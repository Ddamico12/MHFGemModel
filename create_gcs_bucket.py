import subprocess

# Define the bucket name
bucket_name = 'fetus-ultrasound-data'

# Construct the gsutil command to create the bucket
command = ['gsutil', 'mb', f'gs://{bucket_name}']

try:
    subprocess.run(command, check=True)
    print(f"Bucket {bucket_name} created successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error creating bucket: {e}") 