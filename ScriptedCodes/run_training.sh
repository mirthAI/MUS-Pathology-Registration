
#!/bin/bash

# Directories
DATA_DIR="../data/processed_png_data/"
SAVED_MODEL_DIR="./saved_models/"
RESULTS_DIR="./results/"
NUM_OF_FOLDS=6
BATCH_SIZE=1
NUM_OF_EPOCHS_FOR_AFFINE_NETWORK=50
NUM_OF_EPOCHS_FOR_DEFORMABLE_NETWORK=250




# Run the training script with the specified parameters
python train_image_registration_network.py \
    --data_dir "$DATA_DIR" \
    --saved_model_dir "$SAVED_MODEL_DIR" \
    --results_dir "$RESULTS_DIR=" \
    --num_of_folds "$NUM_OF_FOLDS" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs_for_affine_network "$NUM_OF_EPOCHS_FOR_AFFINE_NETWORK" \
    --num_epochs_for_deformable_network "$NUM_OF_EPOCHS_FOR_DEFORMABLE_NETWORK"
