#!/usr/bin/env python3
"""
Cat Detector - Main Entry Point
This script provides a simple CLI to run different versions of the cat detector.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Cat Detector - Choose which version to run')
    parser.add_argument('--version', '-v', choices=['basic', 'safe', 'advanced'],
                        default='advanced', help='Which version of the cat detector to run (default: advanced)')
    
    args = parser.parse_args()
    
    # Map version names to script paths
    version_map = {
        'basic': 'scripts/basic_cat_detector.py',
        'safe': 'scripts/advanced/safe_cat_detector.py',
        'advanced': 'scripts/advanced/advanced_cat_detector.py'
    }
    
    script_path = version_map.get(args.version)
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found.")
        return 1
    
    print(f"Starting {args.version} cat detector...")
    
    # Execute the selected script
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    # Change to the script directory and run it
    current_dir = os.getcwd()
    if script_dir:  # Only change directory if there's a directory component
        os.chdir(script_dir)
    
    try:
        # Execute the script
        os.system(f"python {script_name}")
    finally:
        # Change back to the original directory
        if script_dir:  # Only change back if we changed in the first place
            os.chdir(current_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 