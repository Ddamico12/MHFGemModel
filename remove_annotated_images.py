import os
import shutil
from pathlib import Path

# Define the source and destination paths
source_base = Path('clean-data')
dest_base = Path('clean-data-no-annotations')

# Create the destination directory if it doesn't exist
dest_base.mkdir(exist_ok=True)

# Function to copy non-annotated images
def copy_non_annotated_images(source_folder, dest_folder):
    if not source_folder.exists():
        print(f"Warning: Source folder {source_folder} not found.")
        return
    dest_folder.mkdir(exist_ok=True)
    for item in source_folder.iterdir():
        if item.is_file():
            if 'Annotation' not in item.name:
                shutil.copy2(item, dest_folder / item.name)
        elif item.is_dir():
            copy_non_annotated_images(item, dest_folder / item.name)

# Process each folder in clean-data
for folder in ['train', 'validation', 'test']:
    source_folder = source_base / folder
    dest_folder = dest_base / folder
    print(f"Processing {folder} folder...")
    copy_non_annotated_images(source_folder, dest_folder)
    print(f"{folder} folder processed.")

print("Removal of annotated images complete. Clean data saved in 'clean-data-no-annotations'.") 