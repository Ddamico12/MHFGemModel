import os
import subprocess
from pathlib import Path

# Define the source folder and the GCS bucket
source_folder = Path('clean-data-no-annotations')
bucket_name = 'fetus-ultrasound-data'  # Replace with your actual bucket name

# Function to upload files to GCS
def upload_to_gcs(source_folder, bucket_name):
    if not source_folder.exists():
        print(f"Warning: Source folder {source_folder} not found.")
        return

    # Construct the gsutil command to upload the folder
    command = ['gsutil', '-m', 'cp', '-r', str(source_folder), f'gs://{bucket_name}/']

    try:
        subprocess.run(command, check=True)
        print(f"Uploaded {source_folder} to gs://{bucket_name}/ successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to GCS: {e}")

# Upload the cleaned dataset
upload_to_gcs(source_folder, bucket_name) 