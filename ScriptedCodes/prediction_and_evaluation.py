import argparse
import cv2, os, math
import SimpleITK as sitk
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from scipy import ndimage
import pandas as pd

# User Defined Utilities
from utils.utils_for_evaluation import hist_image_roi, extract_subject_names, preprocess_microus_image, process_slices, computing_metrics
from utils.utils_for_evaluation import preprocess_hist_into_square_images, bounding_box_dimensions, crop_nonzero_region
from utils.utils_for_evaluation import replace_center_with_segmentation, preparing_microus_data_for_registration, preparing_hist_data_for_registration


# from utils.Dataset import ImageRegistrationDataset
from utils.AffineRegistrationModel import AffineNet
from utils.DeformableRegistrationNetwork import DeformRegNet
from utils.SpatialTransformation import SpatialTransformer
from utils.miscellaneous import apply_affine_transformation


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--png_data_dir", type=str, required=True, help="Path to the dataset directory where processed png images are saved")
parser.add_argument("--nifti_data_dir", type=str, required=True, help="Path to the dataset directory where nifti images are saved")
parser.add_argument("--saved_model_dir", type=str, required=True, help="Directory where trained models be saved")
parser.add_argument("--results_dir", type=str, required=True, help="Directory where results be saved")
parser.add_argument("--num_of_folds", type=int, default=6, help="Number of Folds")


args = parser.parse_args()  # Parse command-line arguments


def main(args):
    print("*"*100)
    print(args)
    print("*"*100)
    
    os.makedirs(args.saved_model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = {
        "Image Name": [],
        "Average DSC": [],
        "Average HD": [],
        "Average Urethra Distance": [],
        "Average Landmark 1 Distance": [],
        "Average Landmark 2 Distance": [],
        "Average Landmark 3 Distance": [],
        "Average Landmarks Distance": []
    }
    
    for fold in range(args.num_of_folds):
        print(f"Processing Fold {fold+1}...\n")
        csv_file_path = os.path.join(args.png_data_dir, "Testing_Label_Paths_For_Fold" + str(fold+1) + ".csv")
        # Defining and Loading Trained Affine Model
        path_to_affine_model = os.path.join(args.saved_model_dir, "trained_affine_registration_model_for_Fold" + str(fold+1) + ".pth")
        trained_affine_model = AffineNet().to(device)
        trained_affine_model.load_state_dict(torch.load(path_to_affine_model))

        # Defining and Loading Trained Deformable Registration Network
        path_to_deformable_model = os.path.join(args.saved_model_dir, "trained_deformable_registration_model_for_Fold"+str(fold+1)+".pth")
        trained_deformable_model = DeformRegNet(in_channels=6, out_channels=2, init_features=4).to(device)
        trained_deformable_model.load_state_dict(torch.load(path_to_deformable_model))
        stn = SpatialTransformer()

        subjects = extract_subject_names(csv_file_path)
        for subject in subjects:
            print(f"Processing Case: {subject}")
            result_dir = os.path.join(args.results_dir, subject)
            os.makedirs(result_dir, exist_ok=True)

            source_microus_path = os.path.join(args.nifti_data_dir, subject, "microUS_3D_" + subject + "_image.nii.gz")
            source_hist_path = os.path.join(args.nifti_data_dir, subject, subject + "_histopathology_volume.nii.gz")
            hist_cancer_path = os.path.join(args.nifti_data_dir, subject, subject + "_histopathology_cancer.seg.nrrd")
            hist_landmark_path = os.path.join(args.nifti_data_dir, subject, subject + "_histopathology_landmark.seg.nrrd")
            hist_prostate_path = os.path.join(args.nifti_data_dir, subject, subject + "_histopathology_prostate.seg.nrrd")
            microus_prostate_path = os.path.join(args.nifti_data_dir, subject, subject + "_slice_correspondence.seg.nrrd")
            microus_landmark_path = os.path.join(args.nifti_data_dir, subject, subject + "_microUS_landmark.seg.nrrd")

            hist_cancer_np = sitk.GetArrayFromImage(sitk.ReadImage(hist_cancer_path))
            hist_lmk_np = sitk.GetArrayFromImage(sitk.ReadImage(hist_landmark_path))
            hist_np = sitk.GetArrayFromImage(sitk.ReadImage(source_hist_path))
            hist_prostate_np = sitk.GetArrayFromImage(sitk.ReadImage(hist_prostate_path))


            micro_US_prostate_label = sitk.ReadImage(microus_prostate_path)
            microUS_landmark = sitk.ReadImage(microus_landmark_path)

            microus_volume = sitk.ReadImage(source_microus_path)
            microus_np = sitk.GetArrayFromImage(microus_volume)
            microus_prostate_np = sitk.GetArrayFromImage(micro_US_prostate_label)
            microus_lmk_np = sitk.GetArrayFromImage(microUS_landmark)
            spacing_microus = microus_volume.GetSpacing()

            #### Get US Slice indicies that correspond to histpathology images
            slice_correspondence = []
            for z in range(microus_lmk_np.shape[0]):
                if np.sum(microus_lmk_np[z]) > 0:
                    slice_correspondence.append(z)
            print(f"For Subject: {subject}, following slices have correspondence between histopathology and microus images...\n{slice_correspondence}")

            output_hist_np, output_cancer_np, output_lmk_np, output_prostate_np = process_slices(slice_correspondence, microus_np, microus_prostate_np, hist_np, hist_cancer_np, hist_lmk_np, hist_prostate_np,
                                                                                                 trained_affine_model, trained_deformable_model, apply_affine_transformation, stn, device)

            print('Registration Done... Now Saving...')
            output_hist_volume = sitk.GetImageFromArray(output_hist_np,isVector=True)
            output_cancer_volume = sitk.GetImageFromArray(output_cancer_np,isVector=False)
            output_lmk_volume = sitk.GetImageFromArray(output_lmk_np, isVector=False)
            output_prostate_volume = sitk.GetImageFromArray(output_prostate_np, isVector=False)


            output_hist_volume.SetSpacing((spacing_microus[0]/4,spacing_microus[1]/4,spacing_microus[2]))
            output_cancer_volume.SetSpacing((spacing_microus[0]/4,spacing_microus[1]/4,spacing_microus[2]))
            output_lmk_volume.SetSpacing((spacing_microus[0]/4,spacing_microus[1]/4,spacing_microus[2]))
            output_prostate_volume.SetSpacing((spacing_microus[0]/4,spacing_microus[1]/4,spacing_microus[2]))

            sitk.WriteImage(output_hist_volume, os.path.join(result_dir, subject + "_deformed_hist_volume.nii.gz"), useCompression=True)
            sitk.WriteImage(output_cancer_volume, os.path.join(result_dir, subject + "_deformed_hist_cancer_volume.seg.nrrd"), useCompression=True)
            sitk.WriteImage(output_lmk_volume, os.path.join(result_dir, subject + "_deformed_hist_lmk_volume.seg.nrrd"), useCompression=True)
            sitk.WriteImage(output_prostate_volume, os.path.join(result_dir, subject + "_deformed_hist_prostate_volume.seg.nrrd"), useCompression=True)
            print('')

            print(f'Registered Image for {subject} are saved... now computering metrics...')
            dice_coeff, hausdorff_dist, urethra_dist, landmark1_dist, landmark2_dist, landmark3_dist = computing_metrics(
                micro_US_prostate_label=micro_US_prostate_label,
                deformed_hist_prostate_label=output_prostate_volume, 
                microUS_landmark=microUS_landmark, 
                PATH_landmark_deformed = output_lmk_volume)
            print(dice_coeff, hausdorff_dist,urethra_dist, landmark1_dist, landmark2_dist, landmark3_dist )
            data["Image Name"].append(subject) 
            data["Average DSC"].append(dice_coeff) 
            data["Average HD"].append(hausdorff_dist) 
            data["Average Urethra Distance"].append(urethra_dist) 
            data["Average Landmark 1 Distance"].append(landmark1_dist) 
            data["Average Landmark 2 Distance"].append(landmark2_dist) 
            data["Average Landmark 3 Distance"].append(landmark3_dist) 
            data["Average Landmarks Distance"].append(np.mean([landmark1_dist, landmark2_dist, landmark3_dist]))   

        print('')
        print('-*'*30)
        print('')
    
    df = pd.DataFrame(data)
    output_csv_file = os.path.join(args.results_dir, "metrics.csv")
    df.to_csv(output_csv_file, index=False)

    print(f"Prediction and Evaluation Done!!!\n\n")
    
    

if __name__ == "__main__":
    main(args)
