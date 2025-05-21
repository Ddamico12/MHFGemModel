import os
import kaggle
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle credentials from kaggle.json file."""
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_json_path.exists():
        print("Please download your kaggle.json file from https://www.kaggle.com/settings")
        print("and place it in ~/.kaggle/kaggle.json")
        return False
    
    # Set permissions for kaggle.json
    os.chmod(kaggle_json_path, 0o600)
    return True

def download_dataset():
    """Download the ultrasound fetus dataset."""
    if not setup_kaggle_credentials():
        return
    
    print("Downloading ultrasound fetus dataset...")
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            'orvile/ultrasound-fetus-dataset',
            path='data',
            unzip=True
        )
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_dataset() 