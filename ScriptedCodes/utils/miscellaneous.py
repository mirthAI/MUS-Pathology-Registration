
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients

def apply_affine_transformation(moving_image, theta, mode="bilinear"):
    theta = theta.view(-1, 2, 3)  # Reshape for grid_sample
    grid = torch.nn.functional.affine_grid(theta, moving_image.size(), align_corners=True)
    deformed_image = torch.nn.functional.grid_sample(moving_image, grid, mode=mode, align_corners=True)
    return deformed_image


def smoothness(x_s, y_s):
    alpha = -0.75
    beta = -0.25
    gamma = 0.005
    
    dx_x, dx_y = image_gradients(torch.unsqueeze(x_s, dim=1))
    dy_x, dy_y = image_gradients(torch.unsqueeze(y_s, dim=1))

    dx_x_x, dx_x_y = image_gradients(dx_x)
    dx_y_x, dx_y_y = image_gradients(dx_y)
    dy_x_x, dy_x_y = image_gradients(dy_x)
    dy_y_x, dy_y_y = image_gradients(dy_y)
    
    L1 = torch.sum((alpha*(dx_x_x + dx_y_y) + beta*(dx_x_x + dy_x_y) + gamma*torch.unsqueeze(x_s, dim=1))**2)
    L2 = torch.sum((alpha*(dy_x_x + dy_y_y) + beta*(dx_x_y + dy_y_y) + gamma*torch.unsqueeze(y_s, dim=1))**2)
    
    loss = (L1 + L2) / (512 * 512)
    
    return loss

def loss_function(pred, target, flow, weight=6.0):
    mse_loss = nn.MSELoss()(pred, target)
    smoothness_loss = smoothness(flow[:, 0, :, :], flow[:, 1, :, :]) 
    
    return mse_loss + weight * smoothness_loss  # Weighting the smoothness term

def ssd_loss(fixed_image, deformed_image):
    flattened_fixed = fixed_image.view(-1)
    flattened_deformed = deformed_image.view(-1)
    return torch.nn.functional.mse_loss(flattened_fixed, flattened_deformed)
