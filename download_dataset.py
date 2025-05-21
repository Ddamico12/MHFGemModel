import os
import kagglehub
import shutil

# Set up Kaggle credentials
kaggle_credentials_path = "/Users/danieldamico/Downloads/kaggle.json"
kaggle_dir = os.path.expanduser("~/.kaggle")

# Create .kaggle directory if it doesn't exist
os.makedirs(kaggle_dir, exist_ok=True)

# Copy credentials file to .kaggle directory
shutil.copy2(kaggle_credentials_path, os.path.join(kaggle_dir, "kaggle.json"))

# Set proper permissions for the credentials file
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# Download the dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("orvile/ultrasound-fetus-dataset")
print("Path to dataset files:", path)

# Create a data directory in the current project
project_data_dir = "data"
os.makedirs(project_data_dir, exist_ok=True)

# Copy the dataset to the project directory
source_dir = os.path.join(path, "Ultrasound Fetus Dataset")
source_csv = os.path.join(path, "ultrasound_fetus.csv")

# Copy the dataset directory
if os.path.exists(source_dir):
    shutil.copytree(source_dir, os.path.join(project_data_dir, "Ultrasound Fetus Dataset"), dirs_exist_ok=True)
    print(f"Copied dataset directory to {os.path.join(project_data_dir, 'Ultrasound Fetus Dataset')}")

# Copy the CSV file
if os.path.exists(source_csv):
    shutil.copy2(source_csv, os.path.join(project_data_dir, "ultrasound_fetus.csv"))
    print(f"Copied CSV file to {os.path.join(project_data_dir, 'ultrasound_fetus.csv')}")

print(f"\nDataset is now available in the '{project_data_dir}' directory of your project")

data_dir = "data/Ultrasound Fetus Dataset"
csv_file = "data/ultrasound_fetus.csv" 