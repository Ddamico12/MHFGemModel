import os
import pandas as pd
import shutil
from pathlib import Path
import re

# Define paths
base_path = Path('data/Ultrasound Fetus Dataset')
csv_path = base_path / 'FetusDataset.csv'
datasets_path = base_path / 'Datasets'
output_path = base_path / 'Data/matched_dataset'

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

# Map fetal health classes to categories
# 1.0 = Normal
# 2.0 = Suspect (benign)
# 3.0 = Pathological (malignant)
health_to_category = {
    1.0: 'normal',
    2.0: 'benign',
    3.0: 'malignant'
}

# Create a mapping of image numbers to their categories
image_mapping = {}

# Process each category directory
for category in ['normal', 'benign', 'malignant']:
    category_path = datasets_path / category
    if category_path.exists():
        for img_file in category_path.glob('*.png'):
            try:
                # Skip annotation files
                if '_Annotation' in img_file.name:
                    continue
                    
                # Extract the number from the filename using regex
                match = re.match(r'(\d+)_', img_file.name)
                if match:
                    img_number = int(match.group(1))
                    image_mapping[img_number] = {
                        'category': category,
                        'filename': img_file.name,
                        'has_annotation': (img_file.parent / f"{img_file.stem}_Annotation.png").exists()
                    }
            except (ValueError, AttributeError) as e:
                print(f"Error processing file {img_file.name}: {str(e)}")
                continue

# Create a new DataFrame to store the matched data
matched_data = []

# Iterate through the CSV data
for idx, row in df.iterrows():
    # The index + 1 corresponds to the image number
    img_number = idx + 1
    
    if img_number in image_mapping:
        img_info = image_mapping[img_number]
        expected_category = health_to_category[row['fetal_health']]
        actual_category = img_info['category']
        
        if expected_category != actual_category:
            print(f"Warning: Image {img_number} is in {actual_category} directory but has fetal_health class {row['fetal_health']} (should be in {expected_category})")
        
        source_img = datasets_path / actual_category / img_info['filename']
        
        if source_img.exists():
            # Create category directory in output if it doesn't exist
            category_output = output_path / expected_category
            category_output.mkdir(exist_ok=True)
            
            # Copy the image to the output directory
            dest_img = category_output / source_img.name
            shutil.copy2(source_img, dest_img)
            
            # Copy annotation file if it exists
            if img_info['has_annotation']:
                source_annotation = source_img.parent / f"{source_img.stem}_Annotation.png"
                dest_annotation = category_output / f"{source_img.stem}_Annotation.png"
                shutil.copy2(source_annotation, dest_annotation)
            
            # Add the data to our matched dataset
            matched_data.append({
                'image_number': img_number,
                'image_filename': img_info['filename'],
                'has_annotation': img_info['has_annotation'],
                'original_category': actual_category,
                'corrected_category': expected_category,
                'fetal_health': row['fetal_health'],
                'baseline_value': row['baseline value'],
                'accelerations': row['accelerations'],
                'fetal_movement': row['fetal_movement'],
                'uterine_contractions': row['uterine_contractions'],
                'light_decelerations': row['light_decelerations'],
                'severe_decelerations': row['severe_decelerations'],
                'prolongued_decelerations': row['prolongued_decelerations'],
                'abnormal_short_term_variability': row['abnormal_short_term_variability'],
                'mean_value_of_short_term_variability': row['mean_value_of_short_term_variability'],
                'percentage_of_time_with_abnormal_long_term_variability': row['percentage_of_time_with_abnormal_long_term_variability'],
                'mean_value_of_long_term_variability': row['mean_value_of_long_term_variability'],
                'histogram_width': row['histogram_width'],
                'histogram_min': row['histogram_min'],
                'histogram_max': row['histogram_max'],
                'histogram_number_of_peaks': row['histogram_number_of_peaks'],
                'histogram_number_of_zeroes': row['histogram_number_of_zeroes'],
                'histogram_mode': row['histogram_mode'],
                'histogram_mean': row['histogram_mean'],
                'histogram_median': row['histogram_median'],
                'histogram_variance': row['histogram_variance'],
                'histogram_tendency': row['histogram_tendency']
            })

# Create a DataFrame from the matched data
matched_df = pd.DataFrame(matched_data)

# Save the matched data to a new CSV file
matched_df.to_csv(output_path / 'matched_data.csv', index=False)

# Print summary statistics
print(f"\nTotal images processed: {len(matched_data)}")
print("\nOriginal category distribution:")
print(matched_df['original_category'].value_counts())
print("\nCorrected category distribution:")
print(matched_df['corrected_category'].value_counts())
print("\nFetal health distribution:")
print(matched_df['fetal_health'].value_counts())
print("\nImages with annotations:")
print(matched_df['has_annotation'].value_counts()) 