import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def create_annotated_image(image_path, label, output_path, csv_data):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create a copy of the image for annotation
    annotated_img = img.copy()
    
    # Set ellipse color based on label
    color_map = {
        'BENIGN': (0, 255, 0),      # Green
        'MALIGNANT': (0, 0, 255),   # Red
        'NORMAL': (255, 0, 0)       # Blue
    }
    ellipse_color = color_map.get(label.upper(), (255, 255, 255))
    
    # Get parameters from CSV data
    # Use histogram parameters to determine ellipse size and position
    center_x = int(csv_data['histogram_mean'] * width / 200)  # Scale to image width
    center_y = int(csv_data['histogram_median'] * height / 200)  # Scale to image height
    
    # Use histogram width and variance to determine ellipse axes
    major_axis = int(csv_data['histogram_width'] * width / 200)
    minor_axis = int(csv_data['histogram_variance'] * height / 200)
    
    # Ensure minimum size for visibility
    major_axis = max(major_axis, 20)
    minor_axis = max(minor_axis, 10)
    
    # Ellipse parameters
    center_coordinates = (center_x, center_y)
    axes_length = (major_axis, minor_axis)
    angle = 0
    startAngle = 0
    endAngle = 360
    thickness = 2  # Outline thickness
    
    # Draw the ellipse
    cv2.ellipse(annotated_img, center_coordinates, axes_length, angle, startAngle, endAngle, ellipse_color, thickness)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the annotated image
    cv2.imwrite(str(output_path), annotated_img)

def process_dataset():
    # Set the correct base directory for images
    base_dir = os.path.join("data", "Ultrasound Fetus Dataset", "Ultrasound Fetus Dataset", "Data", "Data")
    datasets_dir = os.path.join(base_dir, "Datasets")
    output_dir = Path("data/annotated_images")
    
    # Read the CSV file
    csv_path = os.path.join(base_dir, "FetusDataset.csv")
    df = pd.read_csv(csv_path)
    
    # Categories to process
    categories = ['benign', 'malignant', 'normal']
    
    # Process each category
    for category in categories:
        input_dir = Path(datasets_dir) / category
        
        if not input_dir.exists():
            print(f"Directory not found: {input_dir}")
            continue
        
        # Create output directory
        output_category_dir = output_dir / category
        os.makedirs(output_category_dir, exist_ok=True)
        
        # Process all images in the directory
        for img_file in input_dir.glob("*_HC.png"):
            # Extract image number from filename (e.g., "1_HC.png" -> 1)
            try:
                img_number = int(img_file.stem.split('_')[0])
                # Get corresponding CSV data for this image
                if img_number <= len(df):
                    csv_row = df.iloc[img_number - 1]  # -1 because CSV is 0-indexed
                    
                    output_path = output_category_dir / f"{img_file.stem}_annotated.png"
                    create_annotated_image(img_file, category.upper(), output_path, csv_row)
                    print(f"Created annotated image: {output_path}")
                else:
                    print(f"No matching CSV data for image {img_file.name}")
            except ValueError:
                print(f"Could not parse image number from {img_file.name}")
                continue

if __name__ == "__main__":
    process_dataset() 