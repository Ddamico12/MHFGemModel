import os
import shutil
from pathlib import Path

# Define the source and destination paths
source_base = Path('data/Ultrasound Fetus Dataset/Ultrasound Fetus Dataset/Data/Data')
dest_base = Path('clean-data')

# Create the destination directory if it doesn't exist
dest_base.mkdir(exist_ok=True)

# Folders to copy
folders = ['train', 'validation', 'test']

for folder in folders:
    source_folder = source_base / folder
    dest_folder = dest_base / folder
    if source_folder.exists():
        print(f"Copying {folder} folder...")
        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
        print(f"{folder} folder copied successfully.")
    else:
        print(f"Warning: {folder} folder not found at {source_folder}")

print("Extraction complete. Folders copied to 'clean-data'.") 