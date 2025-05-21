import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def detect_mask_ellipse(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to better detect the mask
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, binary
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit ellipse to the contour
    if len(largest_contour) >= 5:  # Need at least 5 points to fit an ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse, binary
    return None, binary

def analyze_category(category_path, category_name):
    results = []
    image_files = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(category_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Detect ellipse from mask
        ellipse, binary_mask = detect_mask_ellipse(image)
        
        if ellipse is not None:
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            
            # Calculate contour area and perimeter
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            results.append({
                'image': img_file,
                'category': category_name,
                'center_x': center[0],
                'center_y': center[1],
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'angle': angle,
                'aspect_ratio': major_axis / minor_axis if minor_axis > 0 else 0,
                'contour_area': area,
                'contour_perimeter': perimeter,
                'circularity': 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            })
            
            # Save the binary mask for verification
            output_dir = f"mask_analysis/{category_name}"
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"mask_{img_file}"), binary_mask)
            
            # Draw the detected ellipse on the original image
            result_image = image.copy()
            cv2.ellipse(result_image, ellipse, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"ellipse_{img_file}"), result_image)
    
    return results

def main():
    base_path = "data/Ultrasound Fetus Dataset/OverlayedImages"
    all_results = []
    
    # Create output directory for analysis
    os.makedirs("mask_analysis", exist_ok=True)
    
    # Process each category
    for category in ['normal', 'benign', 'malignant']:
        category_path = os.path.join(base_path, category)
        results = analyze_category(category_path, category)
        all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate statistics for each category
    stats = df.groupby('category').agg({
        'major_axis': ['mean', 'std', 'min', 'max'],
        'minor_axis': ['mean', 'std', 'min', 'max'],
        'angle': ['mean', 'std', 'min', 'max'],
        'aspect_ratio': ['mean', 'std', 'min', 'max'],
        'contour_area': ['mean', 'std', 'min', 'max'],
        'circularity': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Save results
    df.to_csv('ellipse_parameters.csv', index=False)
    stats.to_csv('ellipse_statistics.csv')
    
    # Print summary
    print("\nEllipse Parameters Summary:")
    print("\nCategory-wise Statistics:")
    print(stats)
    
    # Print category counts
    print("\nNumber of images processed per category:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    main() 