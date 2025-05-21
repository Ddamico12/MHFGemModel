import os
import pandas as pd
from pathlib import Path
import re

def extract_image_number(filename):
    """Extract the image number from the filename."""
    match = re.match(r'overlay_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def correlate_metadata(partitioned_dir, original_csv_path, output_dir):
    """
    Correlate partitioned overlays with their metadata from the original CSV.
    
    Args:
        partitioned_dir: Directory containing the partitioned overlays
        original_csv_path: Path to the original FetusDataset.csv
        output_dir: Directory to save the correlated metadata CSVs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the original dataset
    original_df = pd.read_csv(original_csv_path)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_data = []
        split_path = os.path.join(partitioned_dir, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split directory not found: {split_path}")
            continue
        
        # Process each category
        for category in ['normal', 'benign', 'malignant']:
            category_path = os.path.join(split_path, category)
            if not os.path.exists(category_path):
                continue
            
            # Get all overlay images in this category
            for img_file in os.listdir(category_path):
                if not img_file.endswith('.png'):
                    continue
                
                # Extract image number
                img_number = extract_image_number(img_file)
                if img_number is None:
                    print(f"Warning: Could not extract image number from {img_file}")
                    continue
                
                # Get corresponding metadata from original dataset
                # Note: image numbers in the dataset are 1-based, while DataFrame index is 0-based
                if img_number <= len(original_df):
                    metadata = original_df.iloc[img_number - 1].to_dict()
                    metadata['image_filename'] = img_file
                    metadata['category'] = category
                    metadata['split'] = split
                    split_data.append(metadata)
                else:
                    print(f"Warning: Image number {img_number} not found in original dataset")
        
        # Create DataFrame for this split
        split_df = pd.DataFrame(split_data)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'{split}_metadata.csv')
        split_df.to_csv(output_path, index=False)
        
        print(f"\n{split.upper()} Split Summary:")
        print(f"Total images: {len(split_df)}")
        print("\nCategory distribution:")
        print(split_df['category'].value_counts())
        print("\nFetal health distribution:")
        print(split_df['fetal_health'].value_counts())

def main():
    # Define paths
    partitioned_dir = "data/Ultrasound Fetus Dataset/PartitionedElipseOverlays"
    original_csv = "data/Ultrasound Fetus Dataset/FetusDataset.csv"
    output_dir = "data/Ultrasound Fetus Dataset/PartitionedMetadata"
    
    if not os.path.exists(partitioned_dir):
        print(f"Error: Partitioned directory not found at {partitioned_dir}")
        return
    
    if not os.path.exists(original_csv):
        print(f"Error: Original dataset CSV not found at {original_csv}")
        return
    
    print("Starting metadata correlation...")
    correlate_metadata(partitioned_dir, original_csv, output_dir)
    
    print("\nCorrelation complete!")
    print(f"Metadata files have been saved to: {output_dir}")
    print("\nFiles created:")
    for split in ['train', 'val', 'test']:
        print(f"- {split}_metadata.csv")

if __name__ == "__main__":
    main() 