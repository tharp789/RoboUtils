#!/usr/bin/env python3
"""
Script to count files of a certain type in a directory and all subdirectories.
"""

import os
from pathlib import Path
from collections import defaultdict


def count_files_by_type(directory: str, file_extensions: list = None, file_patterns: list = None):
    """
    Count files of specific types in a directory and all subdirectories.
    
    Args:
        directory: Root directory to search
        file_extensions: List of file extensions to search for (e.g., ['.jpg', '.png', '.jpeg'])
                        or single extension string. Extensions can be with or without dot.
                        If None, counts all files
        file_patterns: List of file patterns to search for (e.g., ['*.mcap', '*.jpg'])
                      or single pattern string. Overrides extensions if both provided.
    
    Returns:
        dict: Dictionary with counts per subdirectory, total count, and breakdown by extension
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Normalize extensions to list
    if file_extensions:
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
        # Normalize each extension
        normalized_extensions = []
        for ext in file_extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            normalized_extensions.append(ext.lower())
        file_extensions = normalized_extensions
    
    # Normalize patterns to list
    if file_patterns:
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
        file_patterns = [p.lower() for p in file_patterns]
    
    counts_by_dir = defaultdict(int)
    counts_by_ext = defaultdict(int)
    total_count = 0
    all_files = []
    import fnmatch
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            file_lower = file.lower()
            
            # Check if file matches criteria
            matches = False
            matched_ext = None
            
            if file_patterns:
                # Pattern matching (e.g., '*.mcap', '*.jpg')
                for pattern in file_patterns:
                    if fnmatch.fnmatch(file_lower, pattern):
                        matches = True
                        matched_ext = pattern
                        break
            elif file_extensions:
                # Extension matching
                file_suffix = file_path.suffix.lower()
                if file_suffix in file_extensions:
                    matches = True
                    matched_ext = file_suffix
            else:
                # Count all files
                matches = True
                matched_ext = file_path.suffix.lower() if file_path.suffix else '(no extension)'
            
            if matches:
                counts_by_dir[str(root_path)] += 1
                counts_by_ext[matched_ext] += 1
                total_count += 1
                all_files.append(str(file_path))
    
    return {
        'total': total_count,
        'by_directory': dict(counts_by_dir),
        'by_extension': dict(counts_by_ext),
        'all_files': all_files
    }


def print_summary(results: dict):
    """
    Print a summary of the file count results.
    
    Args:
        results: Results dictionary from count_files_by_type
        show_files: If True, print all file paths
        show_dirs: If True, print counts per directory
        show_extensions: If True, print breakdown by extension/pattern
    """
    print(f"\n{'='*60}")
    print(f"Total files found: {results['total']}")
    print(f"{'='*60}\n")

def main():
    # ============================================
    # CONFIGURATION - Edit these values as needed
    # ============================================
    
    # Directory to search (searches all subdirectories)
    directory = "/home/tyler/Documents/yolo_wire/final_dataset/"
    
    # File types to search for (can be a list or single string)
    # Examples:
    #   file_types = ["jpg", "png", "jpeg"]  # Multiple image types
    #   file_types = "mcap"                   # Single type
    #   file_types = None                     # Count all files
    file_types = ["jpg", "png", "jpeg", "gif", "bmp", "tiff", "webp"]  # Common image formats
        
    try:
        results = count_files_by_type(
            directory,
            file_extensions=file_types
        )
        
        print_summary(results)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

