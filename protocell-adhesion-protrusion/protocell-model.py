"""Protocell on a flat substrate: membrane mechanics, adhesion, protrusion, CG relaxation."""

from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from units import unit

import automembrane.util as u
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.system import System

jax.config.update("jax_enable_x64", True)


# Geometry (verts is (N+1, 2) with verts[-1] == verts[0])

def signed_area(verts: jnp.ndarray) -> jnp.ndarray:
    p = verts[:-1]
    q = jnp.roll(p, -1, axis=0)
    return 0.5 * jnp.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])


def edge_vectors(verts: jnp.ndarray) -> jnp.ndarray:
    p = verts[:-1]
    return jnp.roll(p, -1, axis=0) - p


def edge_lengths(verts: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(edge_vectors(verts), axis=1)


def dual_lengths(verts: jnp.ndarray) -> jnp.ndarray:
    el = edge_lengths(verts)
    return 0.5 * (el + jnp.roll(el, 1))


def outward_normals(verts: jnp.ndarray) -> jnp.ndarray:
    dc = edge_vectors(verts)
    en = jnp.stack([dc[:, 1], -dc[:, 0]], axis=1)
    vn = en + jnp.roll(en, 1, axis=0)
    nrm = jnp.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / jnp.where(nrm > 1e-14, nrm, 1.0)
    vn = vn * jnp.sign(signed_area(verts))
    return jnp.vstack((vn, vn[:1]))


def d_radii(verts: jnp.ndarray) -> jnp.ndarray:
    p = verts[:-1]
    centroid = jnp.mean(p, axis=0)
    return jnp.mean(jnp.linalg.norm(p - centroid, axis=1))


def enclosed_volume(verts: jnp.ndarray) -> jnp.ndarray:
    # Quasi-3D volume (4/3) <r> |A|
    A = jnp.abs(signed_area(verts))
    return (4.0 / 3.0) * d_radii(verts) * A


def smooth_cutoff(d: jnp.ndarray, r_on: float, r_cut: float) -> jnp.ndarray:
    # C2 smootherstep: 1 below r_on, 0 above r_cut
    t = jnp.clip((d - r_on) / (r_cut - r_on), 0.0, 1.0)
    return 1.0 - t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def fold_closing_node(g: jnp.ndarray) -> jnp.ndarray:
    return g.at[0].add(g[-1]).at[-1].set(0.0)


def close_loop(verts: jnp.ndarray) -> jnp.ndarray:
    return verts.at[-1].set(verts[0])


class FlatSubstrate(NamedTuple):
    """Horizontal substrate segment."""
    y0: float = -2.0
    x_min: float = -20.0
    x_max: float = 20.0

    def signed_height(self, verts: jnp.ndarray) -> jnp.ndarray:
        p = verts[:-1]
        dx = jnp.maximum(jnp.maximum(self.x_min - p[:, 0], p[:, 0] - self.x_max), 0.0)
        dy = p[:, 1] - self.y0
        return jnp.sign(dy) * jnp.sqrt(dx * dx + dy * dy)


# Parameter bundles (traced pytrees)

class AdhesionParams(NamedTuple):
    """Adhesion well and steric wall."""
    eps: float = 5.0
    lam: float = 0.02
    d0: float = 0.02
    r_on: float = 0.10
    r_cut: float = 0.20
    k_wall: float = 2.0e4


class OsmoticParams(NamedTuple):
    """Volume penalty."""
    Kv: float = 2.0
    V_bar: float = 1.0


class StretchParams(NamedTuple):
    """Perimeter elasticity."""
    Ksa: float = 0.0
    L0: float = 1.0


class ProtrusionParams(NamedTuple):
    """Protrusive dead load on the leading half."""
    F_anchored: float = 0.0
    mu_slip: float = 1.0
    direction: Tuple[float, float] = (1.0, 0.0)
    normals: Any = None


class ProtoCell(System):
    """Closed 2-D membrane on a flat substrate."""

    def __init__(self, hierarchy, coordinates, materials, substrate, adhesion,
                 osmotic, stretch, protrusion):
        super().__init__(hierarchy, coordinates, materials)
        self.substrate = substrate
        self.adhesion = adhesion
        self.osmotic = osmotic
        self.stretch = stretch
        self.protrusion = protrusion
        self._anchor_proj = []
        self._half_mask = None
        self._gate_Feff = None
        self._build_kernels()

    # Energy kernels

    def _build_kernels(self) -> None:
        material = self.materials[0]
        substrate = self.substrate
        half_fixed = self._half_mask
        gate_F_fixed = self._gate_Feff

        # Bending, tension, edge regularization
        def mechanical(verts):
            comps = material._energy(verts, 0.0)
            return jnp.sum(comps[:3])

        # Osmotic volume
        def osmotic(verts, osm: OsmoticParams):
            V = enclosed_volume(verts)
            r = V / osm.V_bar
            return osm.Kv * (r - jnp.log(r) - 1.0)

        # Adhesion well + steric wall
        def adhesion(verts, adh: AdhesionParams):
            d = substrate.signed_height(verts)
            z = jnp.clip(jnp.maximum(d, 0.0) / adh.lam, 0.0, 30.0)
            well = -adh.eps * jnp.exp(-z)
            w = well * smooth_cutoff(d, adh.r_on, adh.r_cut)
            over = jnp.maximum(adh.d0 - d, 0.0)
            w = w + 0.5 * adh.k_wall * over * over
            return jnp.sum(dual_lengths(verts) * w)

        # Protrusion: equal-share load along the vertex normals
        def protrusion_energy(verts, prot: ProtrusionParams, adh_p: AdhesionParams):
            chi_bound = self._bound_weight_impl(verts, adh_p)
            L_bound = jnp.sum(dual_lengths(verts) * chi_bound)
            f_yield = prot.mu_slip * adh_p.eps / adh_p.lam
            F_hold = f_yield * L_bound
            F_eff = (prot.F_anchored * F_hold
                     / (F_hold + prot.F_anchored + 1e-12))
            if half_fixed is None:
                return 0.0 * jnp.sum(verts)
            w_gate = half_fixed
            nrm = (jax.lax.stop_gradient(outward_normals(verts)[:-1])
                   if prot.normals is None else prot.normals)
            _F = F_eff if gate_F_fixed is None else gate_F_fixed
            n_gate = jnp.maximum(jnp.sum(w_gate), 1e-12)
            f_vert = (_F / n_gate) * w_gate[:, None] * nrm
            return -jnp.sum(f_vert * verts[:-1])

        # Perimeter elasticity
        def stretch(verts, st: StretchParams):
            L = jnp.sum(edge_lengths(verts))
            return 0.5 * st.Ksa * (L - st.L0) ** 2 / st.L0

        def total(verts, adh, osm, st, prot):
            return (mechanical(verts) + osmotic(verts, osm)
                    + adhesion(verts, adh) + stretch(verts, st)
                    + protrusion_energy(verts, prot, adh))

        self._energy_fn = jax.jit(total)
        self._grad_fn = jax.jit(lambda v, a, o, st, pr: fold_closing_node(
            jax.grad(total, argnums=0)(v, a, o, st, pr)))
        self._contact_fn = jax.jit(
            lambda v, adh: jnp.sum(dual_lengths(v) * self._bound_weight_impl(v, adh)))

    def energy(self) -> float:
        v = jnp.asarray(self.coordinates[0])
        return float(self._energy_fn(v, self.adhesion, self.osmotic, self.stretch,
                                     self.protrusion))

    def contact_length(self) -> float:
        v = jnp.asarray(self.coordinates[0])
        return float(self._contact_fn(v, self.adhesion))

    # Bound weight: 0 free, 1 at the well
    def _bound_weight_impl(self, verts, adh: AdhesionParams):
        d = self.substrate.signed_height(verts)
        z = jnp.clip(jnp.maximum(d, 0.0) / adh.lam, 0.0, 30.0)
        density = -adh.eps * jnp.exp(-z) * smooth_cutoff(d, adh.r_on, adh.r_cut)
        return jnp.clip(-density / (adh.eps + 1e-12), 0.0, 1.0)

    # Protocol setup

    def set_leading_half(self) -> None:
        """Leading-half index set [0, N/4] + [N - N/4, N-1]."""
        v = close_loop(jnp.asarray(self.coordinates[0]))
        n = v.shape[0] - 1
        k = n // 4
        self._half_mask = jnp.zeros(n, dtype=v.dtype).at[:k + 1].set(1.0).at[n - k:].set(1.0)
        self._build_kernels()

    def set_rear_pin(self, rear_frac: float = 0.5, bind_frac: float = 0.5) -> int:
        """Freeze the rear fraction of the adhered contact in both components."""
        v = close_loop(jnp.asarray(self.coordinates[0]))
        bound = self._bound_weight_impl(v, self.adhesion) > bind_frac
        d_hat = jnp.asarray(self.protrusion.direction, dtype=v.dtype)
        d_hat = d_hat / jnp.maximum(jnp.linalg.norm(d_hat), 1e-30)
        s_along = v[:-1] @ d_hat
        if float(jnp.sum(bound)) == 0 or rear_frac == 0.0:
            return 0
        s_min = float(jnp.min(jnp.where(bound, s_along, jnp.inf)))
        s_max = float(jnp.max(jnp.where(bound, s_along, -jnp.inf)))
        cut = s_min + rear_frac * (s_max - s_min)
        mask = (bound & (s_along <= cut)).astype(v.dtype)
        if float(jnp.sum(mask)) > 0:
            self._anchor_proj.append(mask)
        return int(jnp.sum(mask))

    def set_protrusion_gate(self) -> dict:
        """Freeze F_eff and set the load normals from the current shape."""
        v = close_loop(jnp.asarray(self.coordinates[0]))
        prot, adh = self.protrusion, self.adhesion
        chi_b = self._bound_weight_impl(v, adh)
        L_bound = jnp.sum(dual_lengths(v) * chi_b)
        f_yield = prot.mu_slip * adh.eps / adh.lam
        F_hold = f_yield * L_bound
        F_eff = prot.F_anchored * F_hold / (F_hold + prot.F_anchored + 1e-12)
        self.protrusion = self.protrusion._replace(normals=outward_normals(v)[:-1])
        self._gate_Feff = F_eff
        self._build_kernels()
        return {"F_hold": float(F_hold), "F_eff": float(F_eff)}

    # Zero the gradient on pinned vertices
    def _project(self, g: jnp.ndarray) -> jnp.ndarray:
        if not self._anchor_proj:
            return g
        free = g[:-1]
        for mask in self._anchor_proj:
            free = free * (1.0 - mask)[:, None]
        return jnp.concatenate([free, g[-1:]])

    # Conjugate-gradient relaxation

    def minimize_cg(self, max_iterations: int = 20000, force_tol: float = 1e-3,
                    grad_stall_window: int = 400, grad_stall_tol: float = 1e-2,
                    lock_growing_contact: bool = False, lock_every: int = 100,
                    lock_bind_frac: float = 0.5, follow_normals: bool = False,
                    normal_every: int = 100, normal_tol_deg: float = 0.5) -> dict:
        """PRP+ CG with Armijo backtracking, growing lock and shape-following load."""
        # Line search and stall constants
        energy_tol = 1e-12
        alpha_init = 1e-4
        alpha_max = 1.0
        armijo_c = 1e-4
        backtrack_rho = 0.5
        max_line_search = 60
        stall_patience = 200

        adh, osm, st = self.adhesion, self.osmotic, self.stretch
        pr = self.protrusion
        verts = close_loop(jnp.asarray(self.coordinates[0]))
        n_dof = int(verts[:-1].size)
        restart_every = n_dof
        shape = verts.shape

        def flat(v):
            return v.reshape(-1)

        def unflat(x):
            return close_loop(x.reshape(shape))

        energy_at = jax.jit(lambda x, p: self._energy_fn(unflat(x), adh, osm, st, p))

        # Tangential lock along the migration axis
        _lock_dir = jnp.asarray(self.protrusion.direction, dtype=verts.dtype)
        _lock_dir = _lock_dir / jnp.maximum(jnp.linalg.norm(_lock_dir), 1e-30)

        def _apply_lock(gv, lock):
            free = gv[:-1]
            comp = (free @ _lock_dir)[:, None] * _lock_dir[None, :]
            return jnp.concatenate([free - lock[:, None] * comp, gv[-1:]])

        grad_at = jax.jit(lambda x, lock, p: flat(_apply_lock(
            self._project(self._grad_fn(unflat(x), adh, osm, st, p)), lock)))
        bound_at = jax.jit(lambda x: self._bound_weight_impl(unflat(x), adh))
        normals_at = jax.jit(lambda x: outward_normals(unflat(x))[:-1])

        load_w = self._half_mask
        follow = bool(follow_normals and pr.normals is not None and load_w is not None)
        loaded = (np.asarray(load_w) > 0) if follow else None

        # Largest angle between load and current normals [deg]
        def stale_deg(p, x_):
            c = np.sum(np.asarray(p.normals) * np.asarray(normals_at(x_)), axis=1)[loaded]
            return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).max())

        lock_mask = jnp.zeros(shape[0] - 1, dtype=verts.dtype)
        if lock_growing_contact:
            lock_mask = (bound_at(flat(verts)) > lock_bind_frac).astype(verts.dtype)

        x = flat(verts)
        g = grad_at(x, lock_mask, pr)
        energy_current = float(energy_at(x, pr))

        force_history = []
        direction, g_prev = None, None
        alpha = alpha_init
        stall_count = 0
        n_lock_updates = 0
        n_normal_updates = 0
        last_refresh = 0
        stop_reason = "max_iterations"
        iteration = 0

        # Re-freeze the load along current normals and restart CG
        def refresh_normals():
            nonlocal pr, g, energy_current, direction, g_prev, stall_count
            nonlocal n_normal_updates, last_refresh
            pr = pr._replace(normals=normals_at(x))
            g = grad_at(x, lock_mask, pr)
            energy_current = float(energy_at(x, pr))
            direction, g_prev, stall_count = None, None, 0
            n_normal_updates += 1
            last_refresh = len(force_history)

        for iteration in range(max_iterations):
            g_max = float(jnp.abs(g).max())
            force_history.append(g_max)

            # Convergence
            if not np.isfinite(g_max):
                stop_reason = "non-finite gradient"
                break
            if g_max < force_tol:
                if follow and stale_deg(pr, x) > normal_tol_deg:
                    refresh_normals()
                    continue
                stop_reason = "force_tol"
                break
            if (grad_stall_window
                    and len(force_history) - last_refresh >= grad_stall_window):
                w = np.asarray(force_history[-grad_stall_window:])
                lvl = float(np.mean(w))
                if lvl > 0 and (float(w.max()) - float(w.min())) / lvl < grad_stall_tol:
                    if follow and stale_deg(pr, x) > normal_tol_deg:
                        refresh_normals()
                        continue
                    stop_reason = (f"|F|max unchanged over {grad_stall_window} steps "
                                   f"(level {lvl:.3e})")
                    break

            # Search direction (PRP+)
            if direction is None or g_prev is None or iteration % restart_every == 0:
                direction = -g
            else:
                numer = jnp.dot(g, g - g_prev)
                denom = jnp.dot(g_prev, g_prev)
                beta = jnp.maximum(0.0, numer / (denom + 1e-30))
                direction = -g + beta * direction
                direction = flat(self._project(direction.reshape(shape)))

            dphi0 = float(jnp.dot(g, direction))
            if dphi0 >= 0:
                direction = -g
                dphi0 = float(jnp.dot(g, direction))
            g_prev = g

            # Armijo backtracking
            def backtrack(a0):
                a = a0
                for _ in range(max_line_search):
                    e = float(energy_at(x + a * direction, pr))
                    if np.isfinite(e) and e <= energy_current + armijo_c * a * dphi0:
                        return True, a, e
                    a *= backtrack_rho
                return False, a, np.inf

            ok, a_try, e_try = backtrack(min(alpha / backtrack_rho, alpha_max))
            if not ok:
                ok, a_try, e_try = backtrack(alpha_max)
            if not ok:
                stop_reason = "line search failed"
                break

            # Step
            alpha = a_try
            x = flat(unflat(x + alpha * direction))
            g = grad_at(x, lock_mask, pr)
            energy_new = e_try
            d_energy = abs(energy_new - energy_current)
            energy_current = energy_new

            # Energy stall
            if d_energy < energy_tol:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= stall_patience and follow and stale_deg(pr, x) > normal_tol_deg:
                refresh_normals()
                continue
            if stall_count >= stall_patience:
                stop_reason = f"stalled (|F|max = {g_max:.3e})"
                break

            # Grow the locked contact
            if lock_growing_contact and iteration % lock_every == 0:
                newly = (bound_at(x) > lock_bind_frac).astype(lock_mask.dtype)
                grown = jnp.maximum(lock_mask, newly)
                if int(jnp.sum(grown - lock_mask)) > 0:
                    lock_mask = grown
                    g = grad_at(x, lock_mask, pr)
                    direction, g_prev = None, None
                    n_lock_updates += 1

            # Follow the shape
            if follow and iteration % normal_every == 0:
                if stale_deg(pr, x) > normal_tol_deg:
                    refresh_normals()

        self.coordinates[0] = np.asarray(unflat(x))
        self.protrusion = pr
        return {
            "stop_reason": stop_reason,
            "iterations": iteration + 1,
            "final_force_max": force_history[-1] if force_history else float("nan"),
            "lock_updates": n_lock_updates,
            "n_locked": int(jnp.sum(lock_mask)),
            "normal_updates": n_normal_updates,
            "normal_stale_deg": stale_deg(pr, x) if follow else float("nan"),
        }


def create_system(*, n_vertices, membrane_radius, substrate_y, initial_gap,
                  adhesion_eps, adhesion_lambda, adhesion_d0, adhesion_r_on,
                  adhesion_r_cut, osmotic_strength, tension, kb_scale, ksl, ksa,
                  target_volume_scale, protrusion_F_anchored, protrusion_mu_slip,
                  initial_x=-15.0, k_wall=2.0e4, substrate_x_min=-20.0,
                  substrate_x_max=20.0) -> ProtoCell:
    """Build a circular protocell resting above a flat substrate."""
    # Moduli
    T = 310 * unit.degK
    KT = (unit.boltzmann_constant * T).to(unit.piconewton * unit.micrometer)
    Kb = (kb_scale * KT).magnitude
    tension_unit = (0.001 * unit.newton / unit.meter).to(
        unit.piconewton / unit.micrometer).magnitude
    Ksg = tension * tension_unit

    # Initial shape
    coord, _ = u.ellipse(n_vertices, R=membrane_radius, x_scale=1.0, y_scale=1.0)
    V0 = float(enclosed_volume(jnp.asarray(coord)))
    coord = coord + np.array([initial_x, substrate_y + membrane_radius + initial_gap])

    material = ClosedPlaneCurveMaterial(Kb=Kb, Ksg=Ksg, Ksl=ksl, Kv=0.0,
                                        V_bar=max(V0 * target_volume_scale, 1e-12))
    L0 = float(np.sum(np.linalg.norm(np.roll(coord[:-1], -1, axis=0) - coord[:-1],
                                     axis=1)))

    return ProtoCell(
        hierarchy={"pm": {}},
        coordinates={"pm": coord},
        materials={"pm": material},
        substrate=FlatSubstrate(y0=substrate_y, x_min=substrate_x_min, x_max=substrate_x_max),
        adhesion=AdhesionParams(eps=adhesion_eps, lam=adhesion_lambda, d0=adhesion_d0,
                                r_on=adhesion_r_on, r_cut=adhesion_r_cut, k_wall=k_wall),
        osmotic=OsmoticParams(Kv=osmotic_strength, V_bar=V0 * target_volume_scale),
        stretch=StretchParams(Ksa=ksa, L0=L0),
        protrusion=ProtrusionParams(F_anchored=protrusion_F_anchored,
                                    mu_slip=protrusion_mu_slip, direction=(1.0, 0.0)),
    )
