#!/bin/bash

# Directories
PNG_DATA_DIR="../data/processed_png_data/"
NIFTI_DATA_DIR="../data/nifti_data/"
SAVED_MODEL_DIR="./saved_models/"
RESULTS_DIR="./results/"
NUM_OF_FOLDS=6





# Run the training script with the specified parameters
python prediction_and_evaluation.py \
    --png_data_dir "$PNG_DATA_DIR" \
    --nifti_data_dir "$NIFTI_DATA_DIR" \
    --saved_model_dir "$SAVED_MODEL_DIR" \
    --results_dir "$RESULTS_DIR=" \
    --num_of_folds "$NUM_OF_FOLDS"
