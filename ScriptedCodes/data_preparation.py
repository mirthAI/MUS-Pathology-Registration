import argparse
import csv, os
from utils.utils_for_data_preparation import converting_nifti_to_png, find_segmentation_dimensions, replace_center_with_segmentation
from utils.utils_for_data_preparation import histopathology_image_preparation, microus_image_preparation, create_training_testing_csv


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, required=True, help="Path to the dataset directory where nifti images are saved")
parser.add_argument("--dest_dir", type=str, required=True, help="Path where the png files will be saved")
args = parser.parse_args()  # Parse command-line arguments

def main(args):
    print("*"*100)
    print(args)
    print("*"*100)
    
    
    print(f"Converting Nifti Image to PNG Formats...\n\n")
    converting_nifti_to_png(args.source_dir, args.dest_dir)
    print(f"Conversion Done!!!\n\n")
    
    source_dir = args.dest_dir
    first_folder_name = os.path.dirname(args.source_dir)
    dest_dir = os.path.join(first_folder_name, "processed_png_data")
    print("Preparing Histopathology PNG Images for Image Registration Tasks...\n\n")
    histopathology_image_preparation(source_dir, dest_dir=dest_dir)
    print(f"Histopathology Preparation Done!!!\n\n")
    
    print("Preparing Micro-US PNG Images for Image Registration Tasks...\n\n")
    microus_image_preparation(source_dir, dest_dir=dest_dir)
    print(f"Micro-US Preparation Done!!!\n\n")
    
    print("Creating CSV file for Six Folds...\n\n")
    create_training_testing_csv(dest_dir, num_leave_out=3)
    print("Folds are Defined and CSV Files are created!!\n\n")

if __name__ == "__main__":
    main(args)
