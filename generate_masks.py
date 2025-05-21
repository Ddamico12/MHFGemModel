import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path

def create_ellipse_overlay(image, ellipse_params):
    """Create an overlay with the ellipse drawn on the original image."""
    overlay = image.copy()
    
    # Draw the ellipse
    cv2.ellipse(overlay,
                (int(ellipse_params['center_x']), int(ellipse_params['center_y'])),
                (int(ellipse_params['axis_x']), int(ellipse_params['axis_y'])),
                ellipse_params['angle'],
                0, 360, (0, 255, 0), 2)  # Green color, 2px thickness
    
    # Blend the overlay with the original image
    result = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    return result

def process_annotations(annotation_dir, output_base_path):
    """Process annotation images and generate overlays."""
    # Create output directory
    os.makedirs(output_base_path, exist_ok=True)
    
    # Initialize counters
    total_processed = 0
    category_counts = {}
    
    # Process each category
    for category in os.listdir(annotation_dir):
        category_path = Path(annotation_dir) / category
        if not category_path.is_dir():
            continue
            
        # Create category output directory
        category_output = Path(output_base_path) / category
        os.makedirs(category_output, exist_ok=True)
        
        print(f"\nProcessing {category} category...")
        category_processed = 0
        
        # Process each annotation image
        for fname in os.listdir(category_path):
            if not fname.endswith("_Annotation.png"):
                continue
                
            # Read annotation image
            img_path = category_path / fname
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Threshold to binary
            _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                print(f"No contours found in: {fname}")
                continue
            
            # Fit ellipse to the largest contour
            cnt = max(contours, key=cv2.contourArea)
            
            if len(cnt) < 5:
                print(f"Not enough points to fit ellipse in: {fname}")
                continue
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis_x, axis_y), angle = ellipse
            
            # Get the original image
            original_img = fname.replace("_Annotation.png", ".png")
            original_path = category_path / original_img
            
            if not original_path.exists():
                print(f"Original image not found: {original_path}")
                continue
            
            # Read original image
            original = cv2.imread(str(original_path))
            if original is None:
                print(f"Could not read original image: {original_path}")
                continue
            
            # Create overlay
            ellipse_params = {
                'center_x': center_x,
                'center_y': center_y,
                'axis_x': axis_x/2,  # OpenCV returns full length, divide by 2 for radius
                'axis_y': axis_y/2,
                'angle': angle
            }
            
            overlay = create_ellipse_overlay(original, ellipse_params)
            
            # Save overlay
            overlay_path = category_output / f"overlay_{original_img}"
            cv2.imwrite(str(overlay_path), overlay)
            
            category_processed += 1
            total_processed += 1
            print(f"Processed: {fname}")
        
        category_counts[category] = category_processed
        print(f"Completed {category}: {category_processed} images processed")
    
    return total_processed, category_counts

def main():
    # Base paths
    annotation_dir = "data/Ultrasound Fetus Dataset/matched_dataset"
    output_base = "data/Ultrasound Fetus Dataset/Overlays"
    
    if not os.path.exists(annotation_dir):
        print(f"Error: Annotation directory not found at {annotation_dir}")
        return
    
    print("Starting overlay generation...")
    total_processed, category_counts = process_annotations(annotation_dir, output_base)
    
    print("\nOverlay generation complete!")
    print(f"Total images processed: {total_processed}")
    print("\nImages processed by category:")
    for category, count in category_counts.items():
        print(f"{category}: {count} images")
    print(f"\nOverlays have been saved to: {output_base}")

if __name__ == "__main__":
    main() 