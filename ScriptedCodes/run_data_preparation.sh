
#!/bin/bash

# Directories
SOURCE_DIR="../data1/nifti_data"         # the dataset directory where nifti images are saved
DESTINATION_DIR="../data1/png_data"          # Directory where png data will be stored

# Run the training script with the specified parameters
python data_preparation.py \
    --source_dir "$SOURCE_DIR" \
    --dest_dir "$DESTINATION_DIR"