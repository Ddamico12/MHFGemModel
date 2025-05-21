Step by step overview:

1. `extract_ellipse_params.py`
- Purpose: Analyzes ultrasound images in the "Overlayed Images" file to extract ellipse parameters
- Key functions:
  - `detect_ellipse`: Detects ellipses in images using OpenCV
  - `analyze_category`: Processes images in each category (normal, benign, malignant)
  - `main`: Orchestrates the analysis and saves results
- Outputs:
  - `ellipse_parameters.csv`: Detailed parameters for each image
  - `ellipse_statistics.csv`: Statistical summaries by category

2. `generate_masks.py`
- Purpose: Creates overlays on original ultrasound images using ellipse parameters
- Key functions:
  - `process_annotations`: Processes annotation images to extract ellipse parameters
  - `create_overlay`: Creates overlays on original images
  - `main`: Manages the overlay generation process
- Outputs:
  - Overlay images saved in `data/Ultrasound Fetus Dataset/Overlays/`

3. `partition_dataset.py`
- Purpose: Splits the dataset into train, validation, and test sets
- Key functions:
  - `partition_dataset`: Splits images while maintaining class balance
  - `main`: Manages the partitioning process
- Features:
  - Uses 70-15-15 split ratio (train-val-test)
  - Maintains class balance across splits
  - Creates summary statistics
- Outputs:
  - Partitioned images in `data/Ultrasound Fetus Dataset/PartitionedElipseOverlays/`
  - Summary CSV with split statistics

4. `correlate_metadata.py`
- Purpose: Matches partitioned overlays with their metadata in the FetusDataset.csv file
- Key functions:
  - `extract_image_number`: Extracts image numbers from filenames
  - `correlate_metadata`: Matches images with metadata from FetusDataset.csv
  - `main`: Manages the correlation process
- Outputs:
  - Separate metadata CSV files for each split (train, val, test)
  - Summary statistics for each split

5. `organize_dataset.py`
- Purpose: Organizes the dataset and matches images with metadata
- Key functions:
  - `organize_dataset`: Matches images with their metadata
  - `main`: Manages the organization process
- Outputs:
  - Organized dataset structure
  - Matched metadata files

The workflow of these scripts is:
1. `extract_ellipse_params.py` analyzes the original images to get ellipse parameters
2. `generate_masks.py` creates overlays using these parameters
3. `partition_dataset.py` splits the overlays into train/val/test sets
4. `correlate_metadata.py` matches the partitioned overlays with their metadata
5. `organize_dataset.py` helps maintain the overall dataset organization


