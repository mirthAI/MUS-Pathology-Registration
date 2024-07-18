
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


def train_affine(affine_model, train_loader, val_loader, optimizer_affine, criterion_affine, device, path_to_save_the_model, num_epochs=10):
    min_loss = float('inf')  # Initialize minimum loss to infinity
    for epoch in range(num_epochs):
        # Initialize tqdm for the training loop
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)
        
        for fixed_image, fixed_mask, moving_image, moving_mask in train_loader_iter:
            ############# AFFINE REGISTRATION ###############################
            optimizer_affine.zero_grad()
            # Pass masks through the network
            fixed_mask = fixed_mask.to(device)
            moving_mask = moving_mask.to(device)
            theta = affine_model((fixed_mask, moving_mask))
            # Apply transformation and compute deformed mask
            deformed_mask = apply_affine_transformation(moving_mask, theta)
            # Calculate SSD loss
            loss_affine = criterion_affine(fixed_mask, deformed_mask)
            # Backpropagate and update weights
            loss_affine.backward()
            optimizer_affine.step()
            
            # Update tqdm description with current loss
            train_loader_iter.set_postfix(loss=loss_affine.item())        
        
        # Save the model if loss decreased
        if loss_affine < min_loss:
            min_loss = loss_affine.item()
            print("Saving model with improved loss:", min_loss)
            torch.save(affine_model.state_dict(), path_to_save_the_model)
            
            
def train_DeformRegNet(affine_model, deformable_model, train_loader, val_loader, optimizer, criterion, device, path_to_save_the_model, results_path, stn, num_epochs=10):
    min_loss = float('inf')  # Initialize minimum loss to infinity    
    for epoch in range(num_epochs):
        # Initialize tqdm for the training loop
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)
        for fixed_image, fixed_mask, moving_image, moving_mask in train_loader_iter:            
            optimizer.zero_grad()
            fixed_mask = fixed_mask.to(device)
            moving_mask = moving_mask.to(device)
            fixed_image = fixed_image.to(device)
            moving_image = moving_image.to(device)
            affine_theta = affine_model((fixed_mask, moving_mask))
            affine_deformed_image = apply_affine_transformation(moving_image, affine_theta, mode="bilinear")
            affine_deformed_mask = apply_affine_transformation(moving_mask, affine_theta)
            input_tensor = torch.cat([affine_deformed_image, fixed_image], dim=1)
            flow = deformable_model(input_tensor)
            registered_img = stn(affine_deformed_image, flow)
            loss_image = criterion(registered_img, fixed_image, flow)
            # For mask
            registered_mask = stn(affine_deformed_mask, flow)
            loss_label = nn.MSELoss()(registered_mask, fixed_mask)
            loss = loss_image + loss_label
            loss.backward()
            optimizer.step()            
            
            # Update tqdm description with current loss
            train_loader_iter.set_postfix(loss=loss.item()) 
        
        # Save the model if loss decreased
        if loss < min_loss:
            min_loss = loss.item()
            print("Saving model with improved loss:", min_loss)
            torch.save(deformable_model.state_dict(), path_to_save_the_model)
            
            # If you want to save intermediate results for debugging, uncomment the following lines
            ######## Saving Results (if you want to save results) ###########
            # save_image(moving_image, f'{results_path}/Epoch_{epoch}_Original_Moving_Images.png')
            # save_image(moving_mask, f'{results_path}/Epoch_{epoch}_Original_Moving_Masks.png')
            # save_image(fixed_image, f'{results_path}/Epoch_{epoch}_Original_Fixed_Images.png')
            # save_image(fixed_mask, f'{results_path}/Epoch_{epoch}_Original_Fixed_Masks.png')
            # save_image(affine_deformed_image, f'{results_path}/Epoch_{epoch}_Affine_Deformed_Moving_Image.png')
            # save_image(registered_img, f'{results_path}/Epoch_{epoch}_Registered_Images.png')    

            # torch.save(theta_tps, f'theta_tps_epoch_{epoch}.pt')
            # torch.save(affine_theta, f'affine_theta_epoch_{epoch}.pt')
