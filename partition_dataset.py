import os
import shutil
import random
from pathlib import Path
import pandas as pd

def partition_dataset(source_dir, output_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partition the dataset into train, validation, and test sets while maintaining class balance.
    
    Args:
        source_dir: Directory containing the overlays organized by class
        output_base: Base directory for the partitioned dataset
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for category in ['normal', 'benign', 'malignant']:
            os.makedirs(os.path.join(output_base, split, category), exist_ok=True)
    
    # Process each category
    for category in ['normal', 'benign', 'malignant']:
        category_path = os.path.join(source_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category directory not found: {category_path}")
            continue
        
        # Get all images in the category
        images = [f for f in os.listdir(category_path) if f.endswith('.png')]
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split the images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to their respective directories
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            for img in split_images:
                src = os.path.join(category_path, img)
                dst = os.path.join(output_base, split_name, category, img)
                shutil.copy2(src, dst)
        
        print(f"\nCategory: {category}")
        print(f"Total images: {n_images}")
        print(f"Train: {len(train_images)} images ({len(train_images)/n_images*100:.1f}%)")
        print(f"Validation: {len(val_images)} images ({len(val_images)/n_images*100:.1f}%)")
        print(f"Test: {len(test_images)} images ({len(test_images)/n_images*100:.1f}%)")

def main():
    # Define paths
    source_dir = "data/Ultrasound Fetus Dataset/Overlays"
    output_base = "data/Ultrasound Fetus Dataset/PartitionedOverlays"
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        return
    
    print("Starting dataset partitioning...")
    print("Using split ratios: 70% train, 15% validation, 15% test")
    partition_dataset(source_dir, output_base)
    
    # Create a summary CSV
    summary_data = []
    for split in ['train', 'val', 'test']:
        for category in ['normal', 'benign', 'malignant']:
            split_path = os.path.join(output_base, split, category)
            if os.path.exists(split_path):
                n_images = len([f for f in os.listdir(split_path) if f.endswith('.png')])
                summary_data.append({
                    'split': split,
                    'category': category,
                    'count': n_images
                })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_base, 'partition_summary.csv'), index=False)
    
    print("\nPartitioning complete!")
    print(f"Partitioned dataset saved to: {output_base}")
    print("\nPartition Summary:")
    print(summary_df.pivot(index='category', columns='split', values='count'))

if __name__ == "__main__":
    main() 