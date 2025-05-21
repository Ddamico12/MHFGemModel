import json
import os
from pathlib import Path
import re

# Define the source folder, GCS bucket path, and output file name
folder_path = 'clean-data-no-annotations/validation'  # Replace with your actual folder path
bucket_path = 'gs://fetus-ultrasound-data/clean-data-no-annotations/validation'  # Replace with your actual bucket path
output_file = 'validation-clean.jsonl'  # Replace with your desired output file name

def natural_sort_key(s):
    # Extract numbers from the filename for sorting
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

# Function to create JSONL examples
def create_jsonl_example(image_path, label, bucket_path):
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": f"{bucket_path}/{image_path}"
                        }
                    },
                    {
                        "text": "You are a diagnostic medical AI trained in fetal neuroimaging.\n\nAnalyze the provided fetal brain ultrasound image and classify it as one of the following:\n– Normal\n– Benign\n– Malignant\n\nBase your classification on the following medically relevant criteria:\n\n    Symmetry and Morphology\n    – Are the brain hemispheres symmetric?\n    – Is the midline intact or shifted?\n    – Are sulci, gyri, and ventricles normal in size and shape for gestational age?\n\n    Lesions or Masses\n    – Are there any focal lesions or space-occupying masses?\n    – Is there evidence of calcification, cystic regions, or hemorrhage?\n\n    Ventricular System\n    – Is there ventriculomegaly, hydrocephalus, or other abnormal dilation?\n    – Are choroid plexuses normal and symmetric?\n\n    Tissue Integrity\n    – Are there areas of hyperechogenicity or hypoechogenicity suggesting necrosis or inflammation?\n\n    Mass Effect or Deformation\n    – Is there compression or displacement of normal brain structures?\n    – Are adjacent tissues affected by any lesion?\n\n    Gestational Appropriateness\n    – Do all visible structures appear appropriate for the estimated gestational age?\n\n\nReturn your analysis as one word, either: normal, benign, malignant"
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": label
                    }
                ]
            }
        ]
    }

# Function to generate JSONL file
def generate_jsonl(folder_path, output_file, bucket_path):
    jsonl_data = []
    for label in ['Normal', 'Benign', 'Malignant']:
        label_path = Path(folder_path) / label
        if label_path.exists():
            # Get all PNG files and sort them naturally
            image_files = sorted(label_path.glob('*.png'), key=natural_sort_key)
            for image_file in image_files:
                image_path = f"{label}/{image_file.name}"
                jsonl_data.append(create_jsonl_example(image_path, label, bucket_path))

    # Write JSONL data to a file
    with open(output_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')

    print(f"JSONL file created successfully at {output_file}.")

# Generate the JSONL file
generate_jsonl(folder_path, output_file, bucket_path) 