import subprocess

# Define the JSONL file and the GCS bucket
jsonl_file = 'validation-clean.jsonl'  # Replace with your actual JSONL file name
bucket_path = 'gs://fetus-ultrasound-data'  # Replace with your actual bucket path

# Construct the gsutil command to upload the JSONL file
command = ['gsutil', 'cp', jsonl_file, f'{bucket_path}/']

try:
    subprocess.run(command, check=True)
    print(f"Uploaded {jsonl_file} to {bucket_path}/ successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error uploading to GCS: {e}") 