"""Protocell probe: run adhesion-prot, snapshots and contact-length plot.

    pixi run python protocell-contact-probe.py --W 1 --F 10
    pixi run python protocell-contact-probe.py --plot
"""
import sys, os, json, argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false")

import numpy as np

# Grid
WS = [0.0, 0.5, 1.0, 2.0]
FS = [0.0, 1.0, 5.0, 10.0]

# Mesh and geometry
N_VERTEX = 60
RADIUS = 1.8
SUBSTRATE_Y = -2.0

# Adhesion
LAMBDA = 0.05
D0 = 0.008
R_CUT = float(os.environ.get("SL_RCUT", 0.05))
R_ON = float(os.environ.get("SL_RON", 0.04))
INITIAL_GAP = min(2.0 * LAMBDA, 0.6 * R_ON)

# Membrane
KV = 200.0
KSA = float(os.environ.get("SL_KSA", 25.0))
KSL = float(os.environ.get("SL_KSL", 2.0))
TENSION = 0.001

# Protocol
BIND_FRAC = 0.5
REAR_FRAC = 0.5
LOCK_EVERY = 100
FOLLOW = int(os.environ.get("SL_FOLLOW", 1))
NORMAL_EVERY = 100
MAX_ITER = int(os.environ.get("SL_ITERS", 100000))
CACHE = os.path.join(HERE, os.environ.get("SL_CACHE", "protocell_probe_cache"))


def mu_for(W):
    return (1.0 * 2.0 / W) * (LAMBDA / 0.05) if W > 0 else 0.0


def run_one(W, F):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from protocell_model import create_system, close_loop

    s = create_system(
        n_vertices=N_VERTEX, membrane_radius=RADIUS,
        adhesion_eps=W, adhesion_lambda=LAMBDA, adhesion_d0=D0,
        adhesion_r_on=R_ON, adhesion_r_cut=R_CUT, initial_gap=INITIAL_GAP,
        substrate_y=SUBSTRATE_Y, osmotic_strength=KV, ksa=KSA, ksl=KSL,
        tension=TENSION, kb_scale=20.0, target_volume_scale=1.0,
        protrusion_F_anchored=0.0, protrusion_mu_frict=mu_for(W))

    # 1. Equilibrate
    s.minimize_cg(max_iterations=MAX_ITER, force_tol=1e-3,
                  grad_stall_window=400, grad_stall_tol=1e-2)
    equil = np.asarray(s.coordinates[0]).copy()
    contact_eq = s.contact_length()

    # 2. Leading half, rear pin, protrusion load
    s.set_leading_half()
    if F > 0.0:
        s.set_rear_pin(rear_frac=REAR_FRAC, bind_frac=BIND_FRAC)
    s.protrusion = s.protrusion._replace(F_anchored=F)
    s.set_protrusion_gate()

    # 3. Protrude
    r1 = s.minimize_cg(max_iterations=MAX_ITER, force_tol=1e-3,
                       grad_stall_window=400, grad_stall_tol=1e-2,
                       lock_growing_contact=(F > 0.0), lock_every=LOCK_EVERY,
                       lock_bind_frac=BIND_FRAC,
                       follow_normals=bool(FOLLOW) and F > 0.0,
                       normal_every=NORMAL_EVERY)
    fin = np.asarray(s.coordinates[0])

    # Contact length
    contact = s.contact_length()

    # Adhered sets for the snapshots
    mask = np.asarray(s._bound_weight_impl(close_loop(jnp.asarray(equil)), s.adhesion)) > BIND_FRAC
    bound = np.asarray(s._bound_weight_impl(close_loop(jnp.asarray(fin)), s.adhesion))

    meta = dict(W=W, F=F, contact_eq=contact_eq, contact=contact, stop_pr=r1["stop_reason"])
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"W{W:g}_F{F:g}.npz"), meta=json.dumps(meta),
             equil=equil, final=fin, mask=mask, bound=bound)
    print(f"W={W:4.1f} F={F:5.1f}  contact {contact_eq:5.2f} -> {contact:5.2f} um  "
          f"{r1['stop_reason'][:20]}", flush=True)
    return meta


def _meta(W, F):
    f = os.path.join(CACHE, f"W{W:g}_F{F:g}.npz")
    if not os.path.exists(f):
        return None
    return json.loads(str(np.load(f, allow_pickle=True)["meta"]))


def plot():
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    C_PM, C_EQ, C_SUB, C_PIN, C_BOUND, C_BAD = (
        "#1f2a4d", "#b8bcc8", "#5aa469", "#e2683c", "#2b7bba", "#c0392b")
    
    dst = os.path.join(HERE, "..", "results", "10_protocell_substrate", "protocell-adhesion")
    os.makedirs(dst, exist_ok=True)

    # Contiguous runs of selected vertices
    def segments(ax, pts, sel, **kw):
        idx = np.where(sel)[0]
        if len(idx):
            for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                ax.plot(pts[seg, 0], pts[seg, 1], **kw)

    # Snapshots
    fig, axes = plt.subplots(len(WS), len(FS), figsize=(3.3 * len(FS), 2.7 * len(WS)), dpi=200)
    for i, W in enumerate(WS):
        for k, F in enumerate(FS):
            ax = axes[i][k]
            f = os.path.join(CACHE, f"W{W:g}_F{F:g}.npz")
            if not os.path.exists(f):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([]); continue
            z = np.load(f, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            eq, fin, mask = z["equil"], z["final"], z["mask"].astype(bool)
            bound_fin = z["bound"] > BIND_FRAC
            rear = np.zeros_like(mask)
            if F > 0 and mask.any():
                xe = eq[:-1, 0]
                lo, hi = xe[mask].min(), xe[mask].max()
                rear = mask & (xe <= lo + REAR_FRAC * (hi - lo))
            cx = fin[:-1, 0].mean()
            ax.add_patch(Rectangle((cx - 3.6, SUBSTRATE_Y - 0.8), 7.2, 0.8,
                                   facecolor=C_SUB, alpha=0.18, edgecolor="none"))
            ax.axhline(SUBSTRATE_Y, color=C_SUB, lw=1.4)
            ax.plot(eq[:, 0], eq[:, 1], color=C_EQ, lw=1.4, ls="--", zorder=2)
            ax.plot(fin[:, 0], fin[:, 1], color=C_PM, lw=1.9, zorder=3)
            segments(ax, fin[:-1], rear, color=C_PIN, lw=5.5, zorder=4, solid_capstyle="butt")
            segments(ax, fin[:-1], bound_fin, color=C_BOUND, lw=2.6, zorder=5, solid_capstyle="butt")
            ax.set_xlim(cx - 3.6, cx + 3.6)
            ax.set_ylim(SUBSTRATE_Y - 0.35, SUBSTRATE_Y + 4.3)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#9a9a9a")
            if i == 0:
                ax.set_title(f"$F$ = {F:g} pN", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"$W$ = {W:g} pN", fontsize=10)
            bad = not m["stop_pr"].startswith("force_tol")
            ax.set_xlabel(f"contact {m['contact_eq']:.2f}$\\rightarrow${m['contact']:.2f} $\\mu$m"
                          + ("\nUNCONVERGED" if bad else ""),
                          fontsize=8, color=C_BAD if bad else "black")
    fig.tight_layout()
    out = os.path.join(CACHE, "protocell-adhesion_gallery.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    shutil.copy2(out, os.path.join(dst, "protocell-adhesion_gallery.png"))

    # Contact length vs adhesion
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    cols = plt.cm.viridis_r(np.linspace(0.12, 0.88, len(FS)))
    for F, col in zip(FS, cols):
        pts = [(m, W) for m, W in ((_meta(W, F), W) for W in WS) if m is not None]
        if pts:
            ax.plot([W for _, W in pts], [m["contact"] for m, _ in pts],
                    "o-", color=col, lw=1.9, ms=6, label=f"{F:g}")
    ax.set_xlabel("Adhesion strength $W$  [pN]", fontsize=10)
    ax.set_ylabel("Contact length [$\\mu$m]", fontsize=10)
    ax.legend(title="protrusive force $F$  [pN]", fontsize=9, title_fontsize=9,
              frameon=False, loc="lower right")
    for sp in ax.spines.values():
        sp.set_color("#9a9a9a")
    ax.tick_params(labelsize=9, color="#9a9a9a")
    fig.tight_layout()
    out2 = os.path.join(CACHE, "protocell-adhesion_contact.png")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    shutil.copy2(out2, os.path.join(dst, "protocell-adhesion_contact.png"))
    print("->", out2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--F", type=float, default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot()
    elif a.F is not None:
        run_one(a.W, a.F)
    else:
        for W in WS:
            for F in FS:
                run_one(W, F)
        plot()
