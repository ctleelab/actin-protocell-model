#!/usr/bin/env python3
"""
Membrane Extension Calculator

This script computes the membrane extension defined as the average distance
from every node on the right half of the 2-D coordinates to the center (0,0)
minus RADIUS of the GUV. It processes all simulation folders found under the given directory.

The script can run for the final frame only or for all frames (time series).
It also supports processing multiple method subfolders (default: arp23, formin,
formin-arp23) under the given save directory.

Usage examples:
  python membrane_extension_filter_by_params.py /path/to/save_dir
  python membrane_extension_filter_by_params.py /path/to/save_dir --time-series --output results.csv

--output can be used to save a CSV with results.

In this example, all the folders have the following pattern: run* or r*. 
This can be changed in find_r0_folders_recursive() if needed.

"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import math
import re
import csv
import sys
from collections import defaultdict

RADIUS = 4.308 # baseline GUV radius in microns, used for extension calculation

def parse_coords(path: str) -> List[Tuple[float, float]]:
        """Parse a text file containing coordinates."""
        coords: List[Tuple[float, float]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
                except ValueError:
                    continue
        return coords

def parse_coords_multi_frame(path: str, max_frames: Optional[int] = None) -> Dict[int, List[Tuple[float, float]]]:
    """
    Parse a multi-frame `space.txt` file. The file format typically contains
    blocks like:

      % frame   N
      % report space { }
      %    class  identity
            cell         1 dynamic_polygon
      x1\ty1
      x2\ty2
      ...
      % end report

    We explicitly look for the '% frame' marker, wait for the 'cell' header,
    then collect coordinate pairs until the '% end report' marker.
    """
    frames: Dict[int, List[Tuple[float, float]]] = {}
    current_frame: Optional[int] = None
    collecting = False
    current_coords: List[Tuple[float, float]] = []

    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Start of a new frame
            if s.lower().startswith('% frame'):
                # Save previous frame if any
                if current_frame is not None and current_coords:
                    frames[current_frame] = current_coords.copy()
                # Reset state for new frame
                m = re.search(r'%\s*frame\s*(\d+)', s, re.IGNORECASE)
                if m:
                    current_frame = int(m.group(1))
                    current_coords = []
                    collecting = False
                    if max_frames is not None and current_frame >= max_frames:
                        # Stop parsing further frames
                        break
                else:
                    current_frame = None
                    collecting = False
                continue

            # End of report for current frame
            if s.lower().startswith('% end report'):
                if current_frame is not None and current_coords:
                    frames[current_frame] = current_coords.copy()
                collecting = False
                current_frame = None
                current_coords = []
                continue

            # Some files put the 'cell' header on a non-comment line (e.g. 'cell 1 dynamic_polygon')
            # If we see a line containing 'cell' and we're inside a frame, start collecting coords.
            if current_frame is not None and not collecting and 'cell' in s.lower():
                collecting = True
                continue

            # Skip other comment lines
            if s.startswith('%'):
                continue

            # If we're in a frame and collecting coordinates, parse numeric lines
            if current_frame is not None and collecting:
                parts = s.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        current_coords.append((x, y))
                    except ValueError:
                        # ignore malformed coordinate lines
                        continue

    return frames

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

def membrane_extension_calculation(coords: List[Tuple[float, float]]) -> float:
    """Calculate the average deviation (distance - RADIUS) for all nodes in the right half of the polygon.
    
    Args:
        coords: List of (x, y) coordinate tuples
        
    Returns:
        Sum of (distance from origin - RADIUS) for all nodes in the right half (leading edge)
        
    Raises:
        ValueError: If no nodes are found in the right half
    """
    # right half indices: all nodes except indices 14..44 (0-based)
    n = len(coords)
    left_start = 14
    left_end_exclusive = 45  # 14..44 inclusive
    right_indices = list(range(0, min(left_start, n))) + list(range(min(left_end_exclusive, n), n))
    if not right_indices:
        raise ValueError('no nodes in right half')
    total = 0.0
    for i in right_indices:

        total += euclidean((0.0, 0.0), coords[i]) - RADIUS
    return total/len(right_indices)

def compute_extension_from_file_final(space_file: Path) -> Optional[float]:
    """Calculate membrane extension (sum of distance - 4.23 for all right half nodes) for final frame.
    
    Args:
        space_file: Path to the space.txt file
        
    Returns:
        Membrane extension value, or None if calculation fails
    """
    try:
        coords = parse_coords(str(space_file))
        if len(coords) < 45:
            # not enough nodes to define halves reliably
            return None
        return membrane_extension_calculation(coords)
    except Exception:
        return None

def compute_extension_from_file_all_frames(space_file: Path, frame_start: int = 0, frame_end: Optional[int] = None, max_frames: Optional[int] = None) -> Dict[int, float]:
    """Compute membrane extension for multiple frames.
    
    Args:
        space_file: Path to the space.txt file
        frame_start: First frame to analyze (inclusive)
        frame_end: Last frame to analyze (inclusive), or None for no limit
        max_frames: Maximum number of frames to parse from file, or None for no limit
        
    Returns:
        Dictionary with frame numbers as keys and membrane extension values as values
    """
    frames = parse_coords_multi_frame(str(space_file), max_frames=max_frames)
    distances: Dict[int, float] = {}
    for frame_num, coords in frames.items():
        if frame_num < frame_start:
            continue
        if frame_end is not None and frame_num > frame_end:
            continue
        if len(coords) < 46:
            continue
        try:
            val = membrane_extension_calculation(coords)
            distances[frame_num] = val
        except Exception:
            continue
    return distances

def find_r0_folders(save_dir: Path) -> List[Path]:
    """Find r0* folders directly under the given directory (non-recursive).

    """
    results = []
    if not save_dir.exists():
        return results
    for item in save_dir.iterdir():
        if item.is_dir() and (item.name.startswith('r0') or item.name.startswith('run')):
            cfg = item / 'config.cym'
            sp = item / 'space.txt'
            if cfg.exists() and sp.exists():
                results.append(item)
    results.sort(key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)
    return results


def find_r0_folders_recursive(root: Path) -> List[Path]:
    """
        Recursively search for simulation output directories that contain both `space.txt` and `config.cym` anywhere under root

    """
    results: List[Path] = []
    if not root.exists():
        return results
    # Use rglob to find directories named run0* at any depth
    for candidate in root.rglob('run*'):
        if candidate.is_dir():
            cfg = candidate / 'config.cym'
            sp = candidate / 'space.txt'
            if cfg.exists() and sp.exists():
                results.append(candidate)

    # De-duplicate and sort numerically by the r0 number when possible
    unique = sorted(set(results), key=lambda p: (str(p.parent), int(p.name[1:]) if p.name[1:].isdigit() else -1))
    return unique

def process_methods(save_dir: Path, methods: List[str], output_csv: Optional[Path], time_series: bool = False, max_frames: Optional[int] = None, frame_start: int = 0, frame_end: Optional[int] = None, avg_last: int = 0):
    all_results = []

    for method in methods:
        # Determine candidate roots to search for run* folders.
        roots_to_search: List[Path] = []
        direct_root = save_dir / method
        if direct_root.exists() and direct_root.is_dir():
            roots_to_search.append(direct_root)
        else:
            # Search for folders named exactly `method` anywhere under save_dir
            for candidate in save_dir.rglob(method):
                if candidate.is_dir():
                    roots_to_search.append(candidate)

        if not roots_to_search and '/' in method:
            maybe = save_dir / method
            if maybe.exists() and maybe.is_dir():
                roots_to_search.append(maybe)

        if not roots_to_search:
            print(f"\n=== Method: {method} (no root found under {save_dir}) ===")
            print(f"Method folder not found: {save_dir / method}")
            continue

        folders: List[Path] = []
        total_r0s = 0
        print(f"\n=== Method: {method} (searching {len(roots_to_search)} root(s)) ===")
        for root in roots_to_search:
            # Prefer direct r0/run children first, then recursive discovery
            r0s = find_r0_folders(root)
            if not r0s:
                r0s = find_r0_folders_recursive(root)
            total_r0s += len(r0s)
            folders.extend(r0s)

        print(f"Found {total_r0s} output folders under method '{method}' (across roots)")

        if not folders:
            print("No folders found for this method.")
            continue

        print(f"Processing {len(folders)} folders:")
        for folder in folders[:5]:  # Show first 5
            print(f"  {folder}")
        if len(folders) > 5:
            print(f"  ... and {len(folders) - 5} more")

        # compute extensions
        method_results = []
        for folder in folders:
            space_file = folder / 'space.txt'
            if time_series:
                vals = compute_extension_from_file_all_frames(space_file, frame_start=frame_start, frame_end=frame_end)
                if avg_last and avg_last > 0:
                    frames_sorted = sorted(vals.keys())
                    if not frames_sorted:
                        continue
                    last_frames = frames_sorted[-avg_last:]
                    vals_list = [vals[f] for f in last_frames if f in vals]
                    if not vals_list:
                        continue
                    mean_val = sum(vals_list) / len(vals_list)
                    rec = {
                        'method': method,
                        'folder': folder.name,
                        'membrane_extension': mean_val,
                    }
                    method_results.append(rec)
                else:
                    for frame, val in sorted(vals.items()):
                        rec = {
                            'method': method,
                            'folder': folder.name,
                            'frame': frame,
                            'membrane_extension': val,
                        }
                        method_results.append(rec)
            else:
                val = compute_extension_from_file_final(folder / 'space.txt')
                rec = {
                    'method': method,
                    'folder': folder.name,
                    'membrane_extension': val,
                }
                method_results.append(rec)

        # print basic summary
        ext_vals = [r['membrane_extension'] for r in method_results if r['membrane_extension'] is not None]
        print(f"Computed {len(method_results)} records for method {method} (valid: {len(ext_vals)})")
        if ext_vals:
            import statistics
            print(f"  mean: {statistics.mean(ext_vals):.6f}, std: {statistics.stdev(ext_vals) if len(ext_vals) > 1 else 0:.6f}, min: {min(ext_vals):.6f}, max: {max(ext_vals):.6f}")

        all_results.extend(method_results)

    # optionally write CSV
    if output_csv and all_results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['method', 'folder', 'membrane_extension']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\nResults written to {output_csv}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Calculate membrane extension for simulation methods')
    parser.add_argument('save-dir', required=True, help='Path containing method subfolders (arp23, formin, formin-arp23)')
    parser.add_argument('--methods', type=str, default='arp23,formin,formin-arp23', help='Comma-separated method folders to process')
    parser.add_argument('--time-series', action='store_true', help='Process all frames (time series)')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames to process for time series')
    parser.add_argument('--frame-start', type=int, default=0, help='Start frame (inclusive) for time-series processing')
    parser.add_argument('--frame-end', type=int, default=None, help='End frame (inclusive) for time-series processing (default: None -> no upper limit)')
    parser.add_argument('--avg-last', type=int, default=0, help='If >0 and --time-series, average over the last N frames and output one value per folder')
    parser.add_argument('--output', '-o', help='Optional CSV output path')

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    results = process_methods(
        save_dir, 
        methods, 
        Path(args.output) if args.output else None,
        time_series=args.time_series,
        max_frames=args.max_frames,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        avg_last=args.avg_last
    )
    if not results:
        print('\nNo results produced (no valid coordinates).')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
