import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
import math
from scipy import ndimage
import SimpleITK as sitk


def hist_image_roi(img, cancer, lmk, prostate):
    gray = img[:,:,0]
    # Find the indices where the value is not 244 (gray border)
    non_gray_indices = np.where(gray != 244)
    # Get the minimum and maximum indices for row and column
    min_row, max_row = np.min(non_gray_indices[0]), np.max(non_gray_indices[0])
    min_col, max_col = np.min(non_gray_indices[1]), np.max(non_gray_indices[1])
    if img.shape[-1] == 3:
        # Extract the region of interest (ROI) without the gray border
        roi = img[min_row:max_row+1, min_col:max_col+1, :]
    else:
        roi = img[min_row:max_row+1, min_col:max_col+1]
    
    return roi, cancer[min_row:max_row+1, min_col:max_col+1], lmk[min_row:max_row+1, min_col:max_col+1], prostate[min_row:max_row+1, min_col:max_col+1]


def extract_subject_names(csv_file_path):
    df = pd.read_csv(csv_file_path, decimal=',')
    # Extract the unique subjects
    unique_subjects = set()
    for path in df['Hist_Mask_Paths']:
        subject_id = path.split('/')[3]
        unique_subjects.add(subject_id)
        
    return unique_subjects


def preprocess_microus_image(image):
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])
    image = Image.fromarray(image)
    image = image_transform(image)
    return image.permute(1,2,0).numpy()


def preprocess_hist_into_square_images(hist_slice,cancer_slice,lmk_slice,hist_prostate_slice):
    # print(f"hist_slice shape: {hist_slice.shape}, cancer_slice shape: {cancer_slice.shape}, lmk_slice shape: {lmk_slice.shape}, hist_prostate_slice shape: {hist_prostate_slice.shape}")
    h,w,_ = hist_slice.shape
    if h < w:
        start = int((w-h)/2)
        hist_pad = 255*np.ones((w,w,3))
        cancer_pad = np.zeros((w,w), dtype=np.uint8)
        lmk_pad = np.zeros((w,w), dtype=np.uint8)
        hist_prostate_pad = np.zeros((w,w), dtype=np.uint8)
        # print(f"hist_pad shape: {hist_pad.shape} | cancer pad shape: {cancer_pad.shape} | lmk_pad shape: {lmk_pad.shape} | hist_prostate_pad shape: {hist_prostate_pad.shape}")
        # print(f"hist_slice shape: {hist_slice.shape} | cancer_slice shape: {cancer_slice.shape} | lmk_slice shape: {lmk_slice.shape} | hist_prostate_slice shape: {hist_prostate_slice.shape}")
        hist_pad[start:start+h, :,:] = hist_slice
        cancer_pad[start:start+h,:] = cancer_slice
        lmk_pad[start:start+h,:] = lmk_slice
        hist_prostate_pad[start:start+h,:] = hist_prostate_slice
    else:
        hist_height, hist_width, _ = hist_slice.shape
        cancer_height, cancer_width = cancer_slice.shape
        if cancer_height != hist_height:
            print('doing resizing')
            cancer_slice = cv2.resize(cancer_slice, (cancer_width, hist_height), interpolation=cv2.INTER_NEAREST)
            hist_prostate_slice = cv2.resize(hist_prostate_slice, (cancer_width, hist_height), interpolation=cv2.INTER_NEAREST)
        start = int((h-w)/2)
        hist_pad = 255*np.ones((h,h,3))
        cancer_pad = np.zeros((h,h)).astype('uint8')
        lmk_pad = np.zeros((h,h)).astype('uint8')
        hist_prostate_pad = np.zeros((h,h), dtype=np.uint8)
        # print(f"hist_pad shape: {hist_pad.shape} | cancer pad shape: {cancer_pad.shape} | lmk_pad shape: {lmk_pad.shape} | hist_prostate_pad shape: {hist_prostate_pad.shape}")
        # print(f"hist_slice shape: {hist_slice.shape} | cancer_slice shape: {cancer_slice.shape} | lmk_slice shape: {lmk_slice.shape} | hist_prostate_slice shape: {hist_prostate_slice.shape}")
        hist_pad[:,start:start+w,:] = hist_slice
        cancer_pad[:,start:start+w] = cancer_slice
        lmk_pad[:,start:start+w] = lmk_slice
        hist_prostate_pad[:,start:start+w] = hist_prostate_slice
    hist_slice = cv2.resize(hist_pad, (512,512))
    cancer_slice = cv2.resize(cancer_pad, (512,512), cv2.INTER_NEAREST)
    cancer_slice = np.repeat(cancer_slice[:, :, np.newaxis], 3, axis=2)
    lmk_slice = cv2.resize(lmk_pad, (512,512), cv2.INTER_NEAREST)
    lmk_slice = np.repeat(lmk_slice[:, :, np.newaxis], 3, axis=2)
    hist_prostate_slice = cv2.resize(hist_prostate_pad, (512,512), cv2.INTER_NEAREST)
    hist_prostate_slice = np.repeat(hist_prostate_slice[:, :, np.newaxis], 3, axis=2)
    return hist_slice/255.0,cancer_slice,lmk_slice, hist_prostate_slice


def bounding_box_dimensions(segmentation):
    # points = np.argwhere(segmentation[:,:,0] != 0)
    # points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    # y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
    # print(f"segmentation shape: {segmentation.shape}")
    non_zero_indices = cv2.findNonZero(segmentation[:, :, 0])
    x, y, w, h = cv2.boundingRect(non_zero_indices)
    return x, y, w, h


def crop_nonzero_region(image, segmentation):
    segmentation_np = np.zeros_like(segmentation)
    segmentation_np[segmentation>0] = 1
    image_np = image
    non_zero_coords = np.argwhere(segmentation_np[:, :, 0] == 1)
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0)
    cropped_img = image_np[y_min:y_max+1, x_min:x_max+1]
    cropped_seg = segmentation_np[y_min:y_max+1, x_min:x_max+1]
    cropped_img *= cropped_seg
    return cropped_img, cropped_seg

def replace_center_with_segmentation(base_image, segmentation_image):
    height, width, _ = base_image.shape
    center_x = width // 2
    center_y = height // 2
    box_width, box_height = segmentation_image.shape[1], segmentation_image.shape[0]
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    end_x = start_x + box_width
    end_y = start_y + box_height
    base_image[start_y:end_y, start_x:end_x, :] = segmentation_image
    return base_image

def preparing_microus_data_for_registration(microus_slice, microus_segmentation):    
    microus_segmentation = microus_segmentation
    
    microus_slice = preprocess_microus_image(microus_slice)
    x, y, w, h = bounding_box_dimensions(microus_segmentation)
    cropped_img_microus, cropped_seg_microus = crop_nonzero_region(microus_slice, microus_segmentation)
    base_image1 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.float32)
    base_image2 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
    new_cropped_img_microus = torch.from_numpy(replace_center_with_segmentation(base_image1, cropped_img_microus)).permute(2, 0, 1)
    new_cropped_seg_microus = torch.from_numpy(replace_center_with_segmentation(base_image2, cropped_seg_microus)).permute(2, 0, 1)
    
    return new_cropped_img_microus.permute(1,2,0).numpy(), new_cropped_seg_microus.permute(1,2,0).numpy(), x, y, w, h


def preparing_hist_data_for_registration(hist_slice, hist_segmentation, cancer_slice, landmark_slice):
    # hist_slice = preprocess_hist_image(hist_slice)
    cropped_img_hist, cropped_seg_hist = crop_nonzero_region(hist_slice, hist_segmentation)
    cropped_cancer_hist, _ = crop_nonzero_region(cancer_slice, hist_segmentation)
    cropped_lmk_hist, _ = crop_nonzero_region(landmark_slice, hist_segmentation)
    x, y, w, h = bounding_box_dimensions(cropped_seg_hist)
    base_image1 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.float32)
    base_image2 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
    base_image3 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)
    base_image4 = np.zeros((round(max(h, w)*1.5), round(max(h, w)*1.5), 3), dtype=np.uint8)

    new_cropped_img_hist = replace_center_with_segmentation(base_image1, cropped_img_hist)
    new_cropped_seg_hist = replace_center_with_segmentation(base_image2, cropped_seg_hist)
    new_cropped_cancer_hist = replace_center_with_segmentation(base_image3, cropped_cancer_hist)
    new_cropped_lmk_hist = replace_center_with_segmentation(base_image4, cropped_lmk_hist)
    return new_cropped_img_hist, new_cropped_seg_hist, new_cropped_cancer_hist, new_cropped_lmk_hist, x, y, w, h   




def process_slices(slice_correspondence, microus_np, microus_prostate_np, hist_np, hist_cancer_np, hist_lmk_np, hist_prostate_np, trained_affine_model, 
                   trained_deformable_model, apply_affine_transformation, stn, device):
    
    # Initializing the output arrays for deformed registered images
    micro_size = microus_np.shape
    output_hist_np = np.zeros((micro_size[0],4*micro_size[1],4*micro_size[2],3), dtype=np.uint8)
    output_prostate_np = np.zeros((micro_size[0],4*micro_size[1],4*micro_size[2]), dtype=np.uint8)
    output_cancer_np = np.zeros((micro_size[0],4*micro_size[1],4*micro_size[2]), dtype=np.uint8)
    output_lmk_np = np.zeros((micro_size[0],4*micro_size[1],4*micro_size[2]), dtype=np.uint8)
    for z in range(len(slice_correspondence)):
        print(f"processing Slice Number: {slice_correspondence[z]}")
        microus_slice = microus_np[slice_correspondence[z],:,:]
        microus_slice = np.repeat(microus_slice[:, :, np.newaxis], 3, axis=2)
        microus_prostate_slice = microus_prostate_np[slice_correspondence[z],:,:]
        microus_prostate_slice = np.repeat(microus_prostate_slice[:, :, np.newaxis], 3, axis=2)
        hist_slice = hist_np[z,:,:,:]
        cancer_slice = hist_cancer_np[z,:,:]
        lmk_slice = hist_lmk_np[z,:,:]
        hist_prostate_slice = hist_prostate_np[z,:,:]
        hist_slice, cancer_slice, lmk_slice, hist_prostate_slice = hist_image_roi(hist_slice, cancer_slice, lmk_slice, hist_prostate_slice)
        hist_slice, cancer_slice, lmk_slice, hist_prostate_slice = preprocess_hist_into_square_images(hist_slice,cancer_slice,lmk_slice, hist_prostate_slice)
        microus_slice_cropped, microus_prostate_slice_cropped,x,y,w,h = preparing_microus_data_for_registration(microus_slice, microus_prostate_slice)
        microus_slice_cropped_resized = cv2.resize(microus_slice_cropped,(512,512))
        microus_prostate_slice_cropped_resized = cv2.resize(microus_prostate_slice_cropped,(512,512), cv2.INTER_NEAREST)
        hist_slice_cropped, hist_prostate_slice_cropped, hist_cancer_slice_cropped, hist_lmk_slice_cropped, _, _, _, _  = preparing_hist_data_for_registration(hist_slice,hist_prostate_slice,cancer_slice,lmk_slice)
        hist_slice_cropped_resized = cv2.resize(hist_slice_cropped,(512,512))
        hist_prostate_slice_cropped_resized = cv2.resize(hist_prostate_slice_cropped,(512,512), cv2.INTER_NEAREST)
        hist_cancer_slice_cropped_resized = cv2.resize(hist_cancer_slice_cropped,(512,512), cv2.INTER_NEAREST)
        hist_lmk_slice_cropped_resized = cv2.resize(hist_lmk_slice_cropped,(512,512), cv2.INTER_NEAREST)

        # hist_slice_cropped_rotated = rotate_image(hist_slice_cropped_resized,rotation_angles[angle_index])
        # hist_prostate_slice_cropped_rotated = rotate_mask(hist_prostate_slice_cropped_resized,rotation_angles[angle_index])
        # hist_cancer_slice_cropped_rotated = rotate_mask(hist_cancer_slice_cropped_resized,rotation_angles[angle_index])
        # hist_lmk_slice_cropped_rotated = rotate_mask(hist_lmk_slice_cropped_resized,rotation_angles[angle_index])
        # angle_index+=1


        fixed_image = torch.from_numpy(microus_slice_cropped_resized).permute(2,0,1).unsqueeze(0)
        fixed_mask = torch.from_numpy(microus_prostate_slice_cropped_resized).permute(2,0,1).unsqueeze(0).to(torch.float32)
        moving_mask = torch.from_numpy(hist_prostate_slice_cropped_resized).permute(2,0,1).unsqueeze(0).to(torch.float32)
        moving_image = torch.from_numpy(hist_slice_cropped_resized).permute(2,0,1).unsqueeze(0).to(torch.float32)
        cancer_slice_tensor = torch.from_numpy(hist_cancer_slice_cropped_resized).permute(2,0,1).unsqueeze(0).to(torch.float32)
        lmk_slice_tensor = torch.from_numpy(hist_lmk_slice_cropped_resized).permute(2,0,1).unsqueeze(0).to(torch.float32)

        trained_affine_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            fixed_mask = fixed_mask.to(device)
            moving_mask = moving_mask.to(device)
            moving_image = moving_image.to(device)
            cancer_slice_tensor = cancer_slice_tensor.to(device)
            lmk_slice_tensor = lmk_slice_tensor.to(device)
            theta = trained_affine_model((fixed_mask, moving_mask))
            affine_deformed_image = apply_affine_transformation(moving_image, theta)
            affine_deformed_prostate = apply_affine_transformation(moving_mask, theta, mode="nearest")
            affine_deformed_cancer = apply_affine_transformation(cancer_slice_tensor, theta, mode="nearest")
            affine_deformed_lmk = apply_affine_transformation(lmk_slice_tensor, theta, mode="nearest")

        trained_deformable_model.eval()
        with torch.no_grad():        
            fixed_image = fixed_image.to(device)
            input_tensor = torch.cat([affine_deformed_image, fixed_image], dim=1)
            flow = trained_deformable_model(input_tensor)
            registered_img = stn(affine_deformed_image, flow)
            registered_prostate = stn(affine_deformed_prostate, flow, mode="nearest")
            registered_landmark_slice = stn(affine_deformed_lmk, flow, mode="nearest")
            registered_cancer_slice = stn(affine_deformed_cancer, flow, mode="nearest")
            
            
        registered_hist_slice = (registered_img[0].permute(1,2,0).detach().cpu().numpy()*255.0).astype('uint8')
        registered_hist_prostate = registered_prostate[0].permute(1,2,0).detach().cpu().numpy()[:,:,0].astype('uint8')
        registered_hist_cancer = registered_cancer_slice[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]
        registered_hist_lmk = registered_landmark_slice[0].permute(1,2,0).detach().cpu().numpy()[:,:,0].astype('uint8')


        new_size = 6*max(w,h)
        registered_hist_slice = cv2.resize(registered_hist_slice, (new_size, new_size))
        registered_hist_prostate = cv2.resize(registered_hist_prostate, (new_size, new_size), cv2.INTER_NEAREST)
        registered_hist_cancer = cv2.resize(registered_hist_cancer, (new_size, new_size), cv2.INTER_NEAREST)
        registered_hist_lmk = cv2.resize(registered_hist_lmk, (new_size, new_size), cv2.INTER_NEAREST)
        print(f"Unique LMK: {np.unique(registered_hist_lmk)}")
        offset_x = int((new_size - 4*w) / 2)
        offset_y = int((new_size - 4*h) / 2)
        output_hist_np[slice_correspondence[z],:,:,:][4*y:4*y+4*h, 4*x:4*x+4*w, :] = registered_hist_slice[offset_y:offset_y+4*h,offset_x:offset_x+4*w,  :]
        output_cancer_np[slice_correspondence[z],:,:][4*y:4*y+4*h, 4*x:4*x+4*w] = registered_hist_cancer[offset_y:offset_y+4*h,offset_x:offset_x+4*w]
        output_lmk_np[slice_correspondence[z],:,:][4*y:4*y+4*h, 4*x:4*x+4*w] = registered_hist_lmk[offset_y:offset_y+4*h,offset_x:offset_x+4*w]
        output_prostate_np[slice_correspondence[z],:,:][4*y:4*y+4*h, 4*x:4*x+4*w] = registered_hist_prostate[offset_y:offset_y+4*h,offset_x:offset_x+4*w]

    return output_hist_np, output_cancer_np, output_lmk_np, output_prostate_np


def keep_largest_connected_component(image, target_label):
    # Create a binary mask for the target label
    binary_mask = (image == target_label)
    
    # Label connected components in the binary mask
    labeled_array, num_features = ndimage.label(binary_mask)
    
    # If no features are found, return the original image
    if num_features == 0:
        return image
    
    # Find the size of each connected component
    component_sizes = np.bincount(labeled_array.ravel())
    
    # Ignore the background component size (component 0)
    component_sizes[0] = 0
    
    # Find the largest connected component
    largest_component = component_sizes.argmax()
    
    # Create a mask for the largest connected component
    largest_component_mask = (labeled_array == largest_component)
    
    # Create a new image where only the largest component is kept
    new_image = np.where(largest_component_mask, target_label, 0)
    
    return new_image

def compute_center_of_mass_distance(US_landmark_array, Hist_landmark_array, slice_correspondence, spacing):
    urethra_dist = []
    landmark1_dist = []
    landmark2_dist = []
    landmark3_dist = []
    for index in slice_correspondence:
        # print(f"Processing Slice: {index}")
        unique_labels = np.unique(US_landmark_array[index])[1:]
        for unique_label in unique_labels:
            # print(f"Processing Unique Label: {unique_label}")
            if unique_label == 1:
                # print('It is Urethra')
                a_hist = np.where(Hist_landmark_array[index] == unique_label, unique_label, 0)
                a_hist = keep_largest_connected_component(a_hist, target_label=unique_label)
                a_microus = np.where(US_landmark_array[index] == unique_label, unique_label, 0)
                a_microus = keep_largest_connected_component(a_microus, target_label=unique_label)
                if np.sum(a_hist)==0:
                    print('The registered landmarks does not have Urethra')
                else:
                    cy_hist, cx_hist = ndimage.center_of_mass(a_hist.astype('uint8'))
                    cy_us, cx_us = ndimage.center_of_mass(a_microus.astype('uint8'))
                    temp = math.dist([cx_hist, cy_hist], [cx_us, cy_us])
                    phy_temp = temp * spacing[0]
                    urethra_dist.append(phy_temp)

            elif unique_label == 2:
                # print('It is Landmark 1')
                a_hist = np.where(Hist_landmark_array[index] == unique_label, unique_label, 0)
                a_hist = keep_largest_connected_component(a_hist, target_label=unique_label)
                a_microus = np.where(US_landmark_array[index] == unique_label, unique_label, 0)
                a_microus = keep_largest_connected_component(a_microus, target_label=unique_label)
                if np.sum(a_hist)==0:
                    print('The registered landmarks does not have Landmark 1')
                else:
                    cy_hist, cx_hist = ndimage.center_of_mass(a_hist.astype('uint8'))
                    cy_us, cx_us = ndimage.center_of_mass(a_microus.astype('uint8'))
                    temp = math.dist([cx_hist, cy_hist], [cx_us, cy_us])
                    phy_temp = temp * spacing[0]
                    landmark1_dist.append(phy_temp)
            elif unique_label == 3:
                # print('It is Landmark 2')
                a_hist = np.where(Hist_landmark_array[index] == unique_label, unique_label, 0)
                a_hist = keep_largest_connected_component(a_hist, target_label=unique_label)
                a_microus = np.where(US_landmark_array[index] == unique_label, unique_label, 0)
                a_microus = keep_largest_connected_component(a_microus, target_label=unique_label)
                if np.sum(a_hist)==0:
                    print('The registered landmarks does not have Landmark 2')
                else:
                    cy_hist, cx_hist = ndimage.center_of_mass(a_hist.astype('uint8'))
                    cy_us, cx_us = ndimage.center_of_mass(a_microus.astype('uint8'))
                    temp = math.dist([cx_hist, cy_hist], [cx_us, cy_us])
                    phy_temp = temp * spacing[0]
                    landmark2_dist.append(phy_temp)

            elif unique_label == 4:
                # print('It is Landmark 3')
                a_hist = np.where(Hist_landmark_array[index] == unique_label, unique_label, 0)
                a_hist = keep_largest_connected_component(a_hist, target_label=unique_label)
                a_microus = np.where(US_landmark_array[index] == unique_label, unique_label, 0)
                a_microus = keep_largest_connected_component(a_microus, target_label=unique_label)
                if np.sum(a_hist)==0:
                    print('The registered landmarks does not have Landmark 3')
                else:
                    cy_hist, cx_hist = ndimage.center_of_mass(a_hist.astype('uint8'))
                    cy_us, cx_us = ndimage.center_of_mass(a_microus.astype('uint8'))
                    temp = math.dist([cx_hist, cy_hist], [cx_us, cy_us])
                    phy_temp = temp * spacing[0]
                    landmark3_dist.append(phy_temp)
            else:
                print(f"The Microus does not have landmark for slice: {index}")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return np.mean(urethra_dist), np.mean(landmark1_dist), np.mean(landmark2_dist), np.mean(landmark3_dist)


def dice_coefficient(US_mask_array, Hist_mask_array, US_indices):
    """
    Calculate Dice coefficient between corresponding masks.

    Args:
    - US_mask_array: Array containing the predicted masks
    - Hist_mask_array: Array containing the ground truth masks
    - US_indices: Indices specifying the pairs of masks to compare

    Returns:
    - List of Dice coefficients for each pair of masks
    """
    dice_coeff = []
    
    for hist_num, US_num in enumerate(US_indices):
        # Ensure the indices are within bounds
        if US_num < len(US_mask_array) and US_num < len(Hist_mask_array):
            predicted_mask = US_mask_array[US_num].astype(bool)
            ground_truth_mask = Hist_mask_array[US_num].astype(bool)
            
            intersection = np.logical_and(predicted_mask, ground_truth_mask)
            
            predicted_sum = predicted_mask.sum()
            ground_truth_sum = ground_truth_mask.sum()
            intersection_sum = intersection.sum()
            
            # Check for division by zero
            if predicted_sum + ground_truth_sum == 0:
                temp = 1.0  # Both masks are empty, consider Dice coefficient as 1
            else:
                temp = 2.0 * (intersection_sum / (predicted_sum + ground_truth_sum))
            
            # Append the Dice coefficient to the list
            dice_coeff.append(temp)
        else:
            print(f"Indices {US_num} out of bounds.")
    
    return np.mean(dice_coeff)


def hausdorff_distance(micro_US_prostate_label, deformed_mask, US_indicies):
    """
    This function will compute the Hausdorff distance between slice and hist labels
    """
    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_dist = []
    for hist_num, US_num in enumerate(US_indicies):
        predicted_mask = sitk.Cast(micro_US_prostate_label[:,:,US_num], sitk.sitkFloat32)
        ground_truth_mask = sitk.Cast(deformed_mask[:,:,US_num], sitk.sitkFloat32)
        hausdorff_filter = sitk.HausdorffDistanceImageFilter()
        hausdorff_filter.Execute(predicted_mask, ground_truth_mask)
        temp = hausdorff_filter.GetHausdorffDistance()
        # print(f'The Hausdorff Distance for slice {hist_num} is {temp}')
        hausdorff_dist.append(temp)
        
    return np.mean(hausdorff_dist)


def computing_metrics(micro_US_prostate_label, deformed_hist_prostate_label, microUS_landmark, PATH_landmark_deformed):
    
    
    # Resampling so that both labels have the same spacing and shape for prostate
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(micro_US_prostate_label)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    deformed_mask = res.Execute(deformed_hist_prostate_label)
    
    # Converting to numpy array
    US_mask_array = sitk.GetArrayFromImage(micro_US_prostate_label)
    Hist_mask_array = sitk.GetArrayFromImage(deformed_mask)
    
    # Resampling so that both labels have the same spacing and shape for lankmarkds
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(microUS_landmark)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    PATH_landmark_deformed = res.Execute(PATH_landmark_deformed)
    # Converting to numpy array
    US_landmark_array = sitk.GetArrayFromImage(microUS_landmark)
    Hist_landmark_array = sitk.GetArrayFromImage(PATH_landmark_deformed)

    

    #### Get US Slice indicies that correspond to histpathology images
    US_indicies = []

    for z in range(US_mask_array.shape[0]):
        if np.sum(US_mask_array[z]) > 0:
            US_indicies.append(z)

    # print(f'Corresponding Slices for {sub} are {US_indicies}')
    dice_coeff = dice_coefficient(US_mask_array, Hist_mask_array, US_indicies)
    # print(f'The Dice Coefficient for {sub} is {dice_coeff}')
    hausdorff_dist = hausdorff_distance(micro_US_prostate_label, deformed_mask, US_indicies)
    # print(f'The Hausdorff Distance for {sub} is {hausdorff_dist}\n\n')
    
    spacing = microUS_landmark.GetSpacing() 
    urethra_dist, landmark1_dist, landmark2_dist,landmark3_dist = compute_center_of_mass_distance(US_landmark_array, Hist_landmark_array, slice_correspondence=US_indicies, spacing=spacing)


    
    return dice_coeff, hausdorff_dist, urethra_dist, landmark1_dist, landmark2_dist, landmark3_dist
