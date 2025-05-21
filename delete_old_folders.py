import shutil
from pathlib import Path

# Define the folders to delete
folders_to_delete = ['clean-data', 'data']

for folder in folders_to_delete:
    folder_path = Path(folder)
    if folder_path.exists():
        print(f"Deleting {folder} folder...")
        shutil.rmtree(folder_path)
        print(f"{folder} folder deleted successfully.")
    else:
        print(f"Warning: {folder} folder not found.")

print("Deletion of old folders complete.") 