#!/usr/bin/env python
import re
import sys
import argparse
import logging
from ccsr_upscale_pipeline.gdrive_utils import GDriveClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("find_gdrive_holes")

def main():
    parser = argparse.ArgumentParser(
        description="Utility to scan a Google Drive folder recursively and identify missing/skipped upscaled tasks."
    )
    parser.add_argument(
        "folder_id",
        nargs="?",
        default="1E25J8gUnHGxuH61KK36CdH1N15gmssh7",
        help="Google Drive Folder ID to scan (default: 1E25J8gUnHGxuH61KK36CdH1N15gmssh7)"
    )
    args = parser.parse_args()

    folder_id = args.folder_id
    logger.info(f"Initializing GDriveClient and scanning folder: {folder_id}")
    
    try:
        client = GDriveClient()
    except Exception as e:
        logger.error(f"Failed to initialize GDriveClient: {e}")
        sys.exit(1)
        
    logger.info("Fetching all files recursively from Google Drive (this may take a moment)...")
    try:
        files = client.list_files_recursively(folder_id)
    except Exception as e:
        logger.error(f"Failed to retrieve files: {e}")
        sys.exit(1)
        
    logger.info(f"Total files found: {len(files)}")
    
    upscaled_by_group = {}
    pattern = re.compile(r"(\d+)_(\d+)_.*_upscaled")
    
    for f in files:
        name = f.get("name", "")
        if "_upscaled" in name:
            match = pattern.match(name)
            if match:
                g = int(match.group(1))
                i = int(match.group(2))
                if g not in upscaled_by_group:
                    upscaled_by_group[g] = set()
                upscaled_by_group[g].add(i)
                
    if not upscaled_by_group:
        logger.warning("No upscaled files matching the format 'GG_II_..._upscaled' were found in the folder.")
        sys.exit(0)
        
    max_group = max(upscaled_by_group.keys())
    logger.info(f"Max group detected: {max_group}")
    
    # Standard group size is 25 based on completed groups (1-25)
    STANDARD_GROUP_SIZE = 25
    missing_tasks = {}
    
    # Iterate from group 1 up to the max group
    for g in range(1, max_group + 1):
        if g not in upscaled_by_group:
            missing_tasks[g] = "all"
            continue
            
        present_images = upscaled_by_group[g]
        
        # For groups before the last group, expect 1..25
        # For the last group, expect 1..max_image_seen in that group
        if g < max_group:
            expected_range = range(1, STANDARD_GROUP_SIZE + 1)
        else:
            max_img_in_last_group = max(present_images)
            expected_range = range(1, max_img_in_last_group + 1)
            
        missing_images = [img for img in expected_range if img not in present_images]
        if missing_images:
            missing_tasks[g] = missing_images
            
    print("\n--- DETECTED HOLES ---")
    if not missing_tasks:
        print("No holes detected! All sequences are complete.")
    else:
        for g in sorted(missing_tasks.keys()):
            val = missing_tasks[g]
            if val == "all":
                print(f"{g}: all")
            else:
                print(f"{g}: {val}")
    print("----------------------\n")

if __name__ == "__main__":
    main()
