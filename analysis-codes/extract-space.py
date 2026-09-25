#!/usr/bin/env python3
"""
Copyright (c) 2026 Honor Akenuwa, Christopher T. Lee, ctleelab

This work is licensed under the Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/

"""


"""
Extract space data from all replicates folders in a save directory.

This script goes through all run* folders in a user-defined save folder and runs
'report space > space.txt' in each folder to extract membrane/polygon coordinates.

In this example, all the folders have the following pattern: run* or r*. 
This can be changed in find_r0_folders() and find_r0_folders_recursive() if needed.

"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional
import sys

def find_r0_folders(save_dir: Path) -> List[Path]:
    """Find all output folders with a particular name pattern in the save directory.
    
    Args:
        save_dir: Path to the save directory
        
    Returns:
        List of paths to output folders, sorted numerically
    """
    if not save_dir.exists():
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")
    
    r0_folders = []
    for item in save_dir.iterdir():
        if item.is_dir() and item.name.startswith('run0'):
            r0_folders.append(item)
    
    # Sort by the numeric part (e.g., r0001, r0002, ...)
    r0_folders.sort(key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 0)
    
    return r0_folders

def find_r0_folders_recursive(root: Path) -> List[Path]:
    """Recursively search for directories named run0* anywhere under root that
    contain both `space.txt` and `config.cym`.

    This allows users to point the script at a top-level project folder and
    have it discover save directories nested under root.
    """
    results: List[Path] = []
    if not root.exists():
        return results
    # Use rglob to find directories named run0* at any depth
    for candidate in root.rglob('run*'):
        if candidate.is_dir():
            # print(candidate)
            cfg = candidate / 'config.cym'
            # sp = candidate / 'space.txt'
            if cfg.exists():
                results.append(candidate)

    # De-duplicate and sort numerically by the r0 number when possible
    unique = sorted(set(results), key=lambda p: (str(p.parent), int(p.name[1:]) if p.name[1:].isdigit() else -1))
    return unique

def run_report_space(folder_path: Path, report_executable: str = "report") -> bool:
    """Run 'report space > space.txt' in the specified folder.
    
    Args:
        folder_path: Path to the output folder containing simulation data
        report_executable: Name or path of the report executable
        
    Returns:
        True if successful, False otherwise
    """
    space_txt_path = folder_path / "space.txt"
    
    # Check if space.txt already exists
    if space_txt_path.exists():
        # os.remove(space_txt_path)
        print(f"  space.txt already exists in {folder_path.name}, skipping...")
        return True
    
    # Check if required files exist
    objects_file = folder_path / "objects.cmo"
    properties_file = folder_path / "properties.cmp"
    
    if not objects_file.exists():
        print(f"  Warning: objects.cmo not found in {folder_path.name}")
        return False
    
    if not properties_file.exists():
        print(f"  Warning: properties.cmp not found in {folder_path.name}")
        return False
    
    # Save original directory
    original_cwd = os.getcwd()
    
    try:
        # Change to the folder directory
        os.chdir(folder_path)
        
        # Run the report command
        print(f"  Running report space in {folder_path.name}...")
        
        with open("space.txt", "w") as output_file:
            result = subprocess.run(
                [report_executable, "space:force"],
                stdout=output_file,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60  # 60 second timeout
            )
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print(f"  ✓ Successfully created space.txt in {folder_path.name}")
            return True
        else:
            print(f"  ✗ Error running report in {folder_path.name}: {result.stderr}")
            # Remove empty or failed output file
            if space_txt_path.exists():
                space_txt_path.unlink()
            return False
            
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        print(f"  ✗ Timeout running report in {folder_path.name}")
        if space_txt_path.exists():
            space_txt_path.unlink()
        return False
    except FileNotFoundError:
        os.chdir(original_cwd)
        print(f"  ✗ Report executable '{report_executable}' not found")
        return False
    except Exception as e:
        os.chdir(original_cwd)
        print(f"  ✗ Unexpected error in {folder_path.name}: {e}")
        if space_txt_path.exists():
            space_txt_path.unlink()
        return False


def extract_space_data_from_save_folder(save_dir: str, report_executable: str = "report") -> dict:
    """Extract space data from all output folders in a save directory.
    
    Args:
        save_dir: Path to the save directory (string)
        report_executable: Name or path of the report executable
        
    Returns:
        Dictionary with statistics about the extraction process
    """
    save_path = Path(save_dir)
    
    print(f"Scanning for folders in: {save_path}")
    
    # Find all r0* folders
    try:
        r0_folders = find_r0_folders_recursive(save_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {"success": False, "error": str(e)}
    
    if not r0_folders:
        print("No output folders found in the save directory")
        return {"success": False, "error": "No output folders found"}
    
    print(f"Found {len(r0_folders)} output folders")
    
    # Statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each folder
    for folder in r0_folders:
        space_txt_path = folder / "space.txt"
        
        if space_txt_path.exists():
            print(f"  space.txt already exists in {folder.name}, skipping...")
            skipped += 1
            continue
            
        if run_report_space(folder, report_executable):
            successful += 1
        else:
            failed += 1
    
    # Summary
    total_processed = successful + failed
    print(f"\nSummary:")
    print(f"  Total folders found: {len(r0_folders)}")
    print(f"  Already had space.txt: {skipped}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total processed: {total_processed}")
    
    return {
        "success": True,
        "total_folders": len(r0_folders),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_processed": total_processed
    }


def process_save_folder(save_dir: str, report_executable: str = "report") -> bool:
    """Simple wrapper function to extract space data from a save folder.
    
    Args:
        save_dir: Path to the save directory containing output folders
        report_executable: Name or path of the report executable (default: "report")
        
    Returns:
        True if all extractions were successful, False if any failed
    """
    result = extract_space_data_from_save_folder(save_dir, report_executable)
    return result.get("success", False) and result.get("failed", 1) == 0


def main():
    """Command line interface for space data extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract space data from all output folders in a save directory"
    )
    parser.add_argument(
        "--save-dir", 
        help="Path to the save directory containing simulation output folders"
    )
    parser.add_argument(
        "--report-executable", 
        default="report",
        help="Name or path of the report executable (default: 'report')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running commands"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        save_path = Path(args.save_dir)
        print(f"DRY RUN: Would scan for simulation output folders in: {save_path}")
        try:
            r0_folders = find_r0_folders(save_path)
            print(f"DRY RUN: Would process {len(r0_folders)} folders:")
            for folder in r0_folders:
                space_txt = folder / "space.txt"
                status = "EXISTS" if space_txt.exists() else "WOULD CREATE"
                print(f"  {folder.name}: {status}")
        except FileNotFoundError as e:
            print(f"DRY RUN ERROR: {e}")
        return 0
    
    # Run the extraction
    result = extract_space_data_from_save_folder(args.save_dir, args.report_executable)
    
    if result["success"]:
        return 0 if result["failed"] == 0 else 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())