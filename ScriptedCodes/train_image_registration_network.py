import argparse
import os
import glob
import csv
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


from PIL import Image
from torchvision import transforms  # Import for image transformations
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import image_gradients
from tqdm import tqdm  # Import tqdm for progress bar
from torchvision.utils import save_image

# User Defined Utilities
from utils.Dataset import ImageRegistrationDataset
from utils.AffineRegistrationModel import AffineNet
from utils.DeformableRegistrationNetwork import DeformRegNet
from utils.SpatialTransformation import SpatialTransformer
from utils.miscellaneous import apply_affine_transformation, smoothness, loss_function, ssd_loss
from utils.utils_for_training import train_affine, train_DeformRegNet


# Define an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory where processed png images are saved")
parser.add_argument("--saved_model_dir", type=str, required=True, help="Directory where trained models be saved")
parser.add_argument("--results_dir", type=str, required=True, help="Directory where results be saved")
parser.add_argument("--num_of_folds", type=int, default=6, help="Number of Folds")
parser.add_argument("--batch_size", type=int, default=1, help="Batch Size")
parser.add_argument("--num_epochs_for_affine_network", type=int, default=30, help="Number of Epochs for Affine Registration Nework")
parser.add_argument("--num_epochs_for_deformable_network", type=int, default=200, help="Number of Epochs for Deformable Registration Nework")

args = parser.parse_args()  # Parse command-line arguments


def main(args):
    print("*"*100)
    print(args)
    print("*"*100)
    
    os.makedirs(args.saved_model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Affine Registration Network For {args.num_of_folds} Folds...\n\n")
    for i in range(args.num_of_folds):
        print(f"Processing Fold {i+1} ...")

        # Dataset Preparation
        train_csv_path_file = os.path.join(args.data_dir, "Training_Label_Paths_For_Fold"+str(i+1)+".csv")
        test_csv_path_file = os.path.join(args.data_dir, "Testing_Label_Paths_For_Fold"+str(i+1)+".csv")

        train_dataset = ImageRegistrationDataset(train_csv_path_file)
        test_dataset = ImageRegistrationDataset(test_csv_path_file)
        print(f"Train Dataset: {len(train_dataset)} | Test Dataset: {len(test_dataset)}")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Defining Model
        affine_model = AffineNet().to(device)
        optimizer_affine = torch.optim.Adam(affine_model.parameters(), lr=0.0001)
        criterion_affine = ssd_loss
        path_to_save_the_model = os.path.join(args.saved_model_dir, "trained_affine_registration_model_for_Fold"+str(i+1)+".pth")
        train_affine(affine_model, train_dataloader, val_dataloader, optimizer_affine, criterion_affine, device, path_to_save_the_model, num_epochs=args.num_epochs_for_affine_network)
        
    print(f"Affine Registration Network Training Done!!!\n\n")
    
    
    print(f"Training Deformable Registration Network For {args.num_of_folds} Folds...\n\n")
    for i in range(args.num_of_folds):
        print(f"Processing Fold {i+1} ...")


        # Loading affine model
        print(f"Let's load Affine Trained Model for Fold {i+1}...")
        path_to_affine_model = os.path.join(args.saved_model_dir, "trained_affine_registration_model_for_Fold" + str(i+1) + ".pth")
        trained_affine_model = AffineNet().to(device)
        trained_affine_model.load_state_dict(torch.load(path_to_affine_model))

        # Results Directory
        results_path = os.path.join(args.results_dir, "Fold" + str(i))
        os.makedirs(results_path, exist_ok=True)

        # Dataset Preparation
        train_csv_path_file = os.path.join(args.data_dir, "Training_Label_Paths_For_Fold"+str(i+1)+".csv")
        test_csv_path_file = os.path.join(args.data_dir, "Testing_Label_Paths_For_Fold"+str(i+1)+".csv")

        train_dataset = ImageRegistrationDataset(train_csv_path_file)
        test_dataset = ImageRegistrationDataset(test_csv_path_file)
        print(f"Train Dataset: {len(train_dataset)} | Test Dataset: {len(test_dataset)}")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Defining Model
        deformable_model = DeformRegNet(in_channels=6, out_channels=2, init_features=4).to(device)
        stn = SpatialTransformer()
        optimizer = torch.optim.Adam(deformable_model.parameters(), lr=0.01)
        criterion = loss_function    
        path_to_save_deformable_model = os.path.join(args.saved_model_dir, "trained_deformable_registration_model_for_Fold"+str(i+1)+".pth")
        train_DeformRegNet(affine_model=trained_affine_model, 
                           deformable_model=deformable_model, 
                           train_loader=train_dataloader, 
                           val_loader=val_dataloader, 
                           optimizer=optimizer, 
                           criterion=criterion, 
                           device=device, 
                           path_to_save_the_model=path_to_save_deformable_model, 
                           results_path=results_path, 
                           stn=stn,
                           num_epochs=args.num_epochs_for_deformable_network)
        print('')
        print('--'*90)
        print('')
        
    print(f"Deformable Registration Network Training Done!!!\n\n")
    
    

if __name__ == "__main__":
    main(args)
