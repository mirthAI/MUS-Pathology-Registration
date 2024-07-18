
import os
import glob
import csv
import SimpleITK as sitk
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

def converting_nifti_to_png(source_dir, dest_dir):
    """
    Convert NIfTI and NRRD files to PNG format.

    This function reads medical imaging data in NIfTI (.nii.gz) and NRRD (.seg.nrrd) formats,
    extracts relevant slices, and saves them as PNG files in the specified destination directory.
    It also creates a CSV file listing the generated PNG files.

    Parameters:
    - source_dir (str): Path to the source directory containing subfolders of NIfTI and NRRD files.
    - dest_dir (str): Path to the destination directory where PNG files will be saved.

    The CSV file generated will contain columns: 
    - Micro-US Image Names
    - Micro-US Masks Names
    - PATH Image Names
    - PATH Mask Names
    """
    
    # Get the list of source images
    source_images = sorted(os.listdir(source_dir))

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Create a CSV file to save the file names
    csv_file_path = "./file_names.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Micro-US Image Names", "Micro-US Masks Names", "PATH Image Names", "PATH Mask Names"])

        for folder in source_images:
            dest_folder = os.path.join(dest_dir, folder)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            micro_image_path = glob.glob(os.path.join(source_dir, folder, "micro*image.nii.gz"))[0]
            micro_mask_path = glob.glob(os.path.join(source_dir, folder, "*correspondence*.seg.nrrd"))[0]
            hist_image_path = glob.glob(os.path.join(source_dir, folder, "*hist*.nii.gz"))[0]
            hist_mask_path = glob.glob(os.path.join(source_dir, folder, "*prostate.seg.nrrd"))[0]

            micro_image = sitk.GetArrayFromImage(sitk.ReadImage(micro_image_path))
            micro_mask = sitk.GetArrayFromImage(sitk.ReadImage(micro_mask_path))
            hist_image = sitk.GetArrayFromImage(sitk.ReadImage(hist_image_path))
            hist_mask = sitk.GetArrayFromImage(sitk.ReadImage(hist_mask_path))
            number_of_nonzero_slices = 0
            for i in tqdm(range(micro_image.shape[0]), desc=f"Processing Subject: {folder} | Slice Number", unit="slice"):
                temp = np.sum(micro_mask[i, :, :])
                if temp != 0:
                    image = micro_image[i, :, :]
                    mask = micro_mask[i, :, :] * 255
                    dest_micro_image_path = os.path.join(dest_folder, f"{folder}_microus_slice_{i}.png")
                    dest_micro_mask_path = os.path.join(dest_folder, f"{folder}_microus_label_{i}.png")
                    cv2.imwrite(dest_micro_image_path, image)
                    cv2.imwrite(dest_micro_mask_path, mask.astype(np.uint8))

                    temp_hist_image = hist_image[number_of_nonzero_slices, :, :, :].astype(np.uint8)
                    temp_hist_image = cv2.cvtColor(temp_hist_image, cv2.COLOR_RGB2BGR)
                    temp_hist_mask = hist_mask[number_of_nonzero_slices, :, :] * 255
                    dest_hist_image_path = os.path.join(dest_folder, f"{folder}_hist_slice_{i}.png")
                    dest_hist_mask_path = os.path.join(dest_folder, f"{folder}_hist_label_{i}.png")
                    cv2.imwrite(dest_hist_image_path, temp_hist_image.astype(np.uint8))
                    cv2.imwrite(dest_hist_mask_path, temp_hist_mask.astype(np.uint8))

                    number_of_nonzero_slices += 1
                    writer.writerow([dest_micro_image_path, dest_micro_mask_path, dest_hist_image_path, dest_hist_mask_path])

            print(f"For {folder}, Slice Correspondance Between Histopathology and Micro-US Exists for {number_of_nonzero_slices} Slices...")
    print('All done!!!')


def find_segmentation_dimensions(segmentation_file):
    # Read the segmentation file
    segmentation_image = cv2.imread(segmentation_file, cv2.IMREAD_GRAYSCALE)
    
    # Find all non-zero pixels
    non_zero_indices = cv2.findNonZero(segmentation_image)
    
    if non_zero_indices is None:
        return 0, 0  # If no non-zero pixels found, return 0 for both width and height
    
    # Find the bounding rectangle around non-zero pixels
    x, y, w, h = cv2.boundingRect(non_zero_indices)
    
    return w, h

def replace_center_with_segmentation(base_image, segmentation_image):
    # Get dimensions of base image
    height, width, _ = base_image.shape
    
    # Find the center of the base image
    center_x = width // 2
    center_y = height // 2
    
    # Find the starting and ending coordinates for the boundary box
    box_width, box_height = segmentation_image.shape[1], segmentation_image.shape[0]
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    end_x = start_x + box_width
    end_y = start_y + box_height
    
    # Replace the center of the base image with the segmentation image
    base_image[start_y:end_y, start_x:end_x, :] = segmentation_image
    
    return base_image


def histopathology_image_preparation(source_dir, dest_dir):    
    subfolders = sorted(os.listdir(source_dir))
    hist_coordinates = []
    for i in range(len(subfolders)):
        new_folder = os.path.join(dest_dir, subfolders[i])
        os.makedirs(new_folder, exist_ok=True)
        hist_images_paths = sorted(glob.glob(os.path.join(source_dir, subfolders[i], "*_hist_slice_*")))
        hist_mask_paths = sorted(glob.glob(os.path.join(source_dir, subfolders[i], "*_hist_label_*")))
        for hist_image_path in tqdm(hist_images_paths, desc=f'Processing Histopathology Case: {subfolders[i]}', unit='image'):
            hist_mask_path = hist_image_path.replace('slice', 'label')
            hist_filename = hist_image_path.split("/")[-1]
            hist_seg_filename = hist_mask_path.split("/")[-1]
            img_hist = cv2.imread(hist_image_path, cv2.IMREAD_COLOR)
            seg_hist = cv2.imread(hist_mask_path, cv2.IMREAD_COLOR) 
            if img_hist is not None:    
                
                # Find all non-zero pixel coordinates
                seg_hist[np.where(seg_hist==254)] = 255
                non_zero_coords = np.argwhere(seg_hist[:,:,0] == 255)
                # Crop the image to the non-zero region
                y_min, x_min = non_zero_coords.min(axis=0)
                y_max, x_max = non_zero_coords.max(axis=0)
                cropped_img_hist = img_hist[y_min:y_max+1, x_min:x_max+1]
                cropped_seg_hist = seg_hist[y_min:y_max+1, x_min:x_max+1]
                cropped_img_hist = cropped_img_hist * (cropped_seg_hist/255.0).astype('uint8')        
                ##############################################################################
                w, h = find_segmentation_dimensions(hist_mask_path)
                base_image1 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
                base_image2 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
                new_cropped_img_hist = replace_center_with_segmentation(base_image1, cropped_img_hist)
                new_cropped_seg_hist = replace_center_with_segmentation(base_image2, cropped_seg_hist)
                ############################################################################

                # # Save the processed image
                hist_image_path_new = os.path.join(new_folder, hist_filename)
                hist_mask_path_new = os.path.join(new_folder, hist_seg_filename)
                cv2.imwrite(hist_image_path_new, new_cropped_img_hist)
                cv2.imwrite(hist_mask_path_new, new_cropped_seg_hist)
    print('All Histopathology Image are Prepared for Image Registration Task')
    
    
def microus_image_preparation(source_dir, dest_dir):  
    subfolders = sorted(os.listdir(source_dir))
    microus_coordinates = []
    for i in range(len(subfolders)):
        new_folder = os.path.join(dest_dir, subfolders[i])
        os.makedirs(new_folder, exist_ok=True)
        microus_images_paths = sorted(glob.glob(os.path.join(source_dir, subfolders[i], "*_microus_slice_*")))
        microus_mask_paths = sorted(glob.glob(os.path.join(source_dir, subfolders[i], "*_microus_label_*")))
        for microus_image_path in tqdm(microus_images_paths, desc=f'Processing Micro-US Case: {subfolders[i]}', unit='image'):
            microus_mask_path = microus_image_path.replace('slice', 'label')
            microus_filename = microus_image_path.split("/")[-1]
            microus_seg_filename = microus_mask_path.split("/")[-1]
            img_microus = cv2.imread(microus_image_path, cv2.IMREAD_COLOR)
            seg_microus = cv2.imread(microus_mask_path, cv2.IMREAD_COLOR) 
            if img_microus is not None:    
                # print(f"Processing ... {microus_filename}")
                # Find all non-zero pixel coordinates
                seg_microus[np.where(seg_microus==254)] = 255
                non_zero_coords = np.argwhere(seg_microus[:,:,0] == 255)
                # Crop the image to the non-zero region
                y_min, x_min = non_zero_coords.min(axis=0)
                y_max, x_max = non_zero_coords.max(axis=0)
                cropped_img_microus = img_microus[y_min:y_max+1, x_min:x_max+1]
                cropped_seg_microus = seg_microus[y_min:y_max+1, x_min:x_max+1]
                ##############################################################################
                # cropped_img_microus = cv2.cvtColor(cropped_img_microus, cv2.COLOR_BGR2RGB)
                cropped_img_microus = cropped_img_microus * (cropped_seg_microus/255.0).astype('uint8')
                w, h = find_segmentation_dimensions(microus_mask_path)
                # print(f"Width: {w} | Height: {h}")
                base_image1 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
                base_image2 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
                new_cropped_img_microus = replace_center_with_segmentation(base_image1, cropped_img_microus)
                new_cropped_seg_microus = replace_center_with_segmentation(base_image2, cropped_seg_microus)
                ############################################################################
                # print(f"new_cropped_img_microus: {new_cropped_img_microus.shape} | new_cropped_seg_microus: {new_cropped_seg_microus.shape}")
                # # Save the processed image
                microus_image_path_new = os.path.join(new_folder, microus_filename)
                microus_mask_path_new = os.path.join(new_folder, microus_seg_filename)
                cv2.imwrite(microus_image_path_new, new_cropped_img_microus)
                cv2.imwrite(microus_mask_path_new, new_cropped_seg_microus)
    print('MicroUS Processing Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    

def create_training_testing_csv(dest_dir, num_leave_out=3):
    # List all subfolders
    subfolders = sorted([f.path for f in os.scandir(dest_dir) if f.is_dir()])

    # Total number of cycles
    num_folds = len(subfolders) // num_leave_out

    # Iterate over each cycle
    for cycle in tqdm(range(num_folds), desc="Folds"):
        # Calculate the starting index for the subfolders to leave out
        start_idx = len(subfolders) - (cycle + 1) * num_leave_out
        # Calculate the ending index for the subfolders to leave out
        end_idx = start_idx + num_leave_out
        
        # Get the list of subfolders for this cycle
        selected_subfolders = subfolders[:start_idx] + subfolders[end_idx:]
        leave_out_subfolders = subfolders[start_idx:end_idx]
        
        training_record = []
        testing_record = []
        training_csv_file = os.path.join(dest_dir, f"Training_Label_Paths_For_Fold{cycle + 1}.csv")
        testing_csv_file = os.path.join(dest_dir, f"Testing_Label_Paths_For_Fold{cycle + 1}.csv")
        
        # Training Lists
        for subfolder in selected_subfolders:
            microus_mask_paths = sorted(glob.glob(os.path.join(subfolder, "*_microus_label_*")))
            hist_mask_paths = sorted(glob.glob(os.path.join(subfolder, "*_hist_label_*")))
            for i in range(len(microus_mask_paths)):
                training_record.append({"Hist_Mask_Paths": hist_mask_paths[i], "Micour US Maks": microus_mask_paths[i]})    
        df = pd.DataFrame(training_record)
        df.to_csv(training_csv_file, index=False)
        
        # Testing Lists
        for subfolder in leave_out_subfolders:
            microus_mask_paths = sorted(glob.glob(os.path.join(subfolder, "*_microus_label_*")))
            hist_mask_paths = sorted(glob.glob(os.path.join(subfolder, "*_hist_label_*")))
            for i in range(len(microus_mask_paths)):
                testing_record.append({"Hist_Mask_Paths": hist_mask_paths[i], "Micour US Maks": microus_mask_paths[i]})
        
        df = pd.DataFrame(testing_record)
        df.to_csv(testing_csv_file, index=False)
    
    print("Processing completed.")


