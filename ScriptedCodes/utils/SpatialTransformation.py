
import torch
import torch.nn as nn
import torch.nn.functional as F


### Utilities for Deformable Registration
class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, image, flow, mode="bilinear"):
        """
        image: (B, C, H, W)
        flow: (B, 2, H, W), flow[i, j, h, w] gives the displacements (dh, dw)
        """
        B, C, H, W = image.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if image.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flow

        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # The following lines will not work if batch size is not equal to 1.
        output0 = nn.functional.grid_sample(image[0][0].unsqueeze(0).unsqueeze(0), vgrid.permute(0, 2, 3, 1), mode=mode, padding_mode='border', align_corners=True)
        output1 = nn.functional.grid_sample(image[0][1].unsqueeze(0).unsqueeze(0), vgrid.permute(0, 2, 3, 1), mode=mode, padding_mode='border', align_corners=True)
        output2 = nn.functional.grid_sample(image[0][2].unsqueeze(0).unsqueeze(0), vgrid.permute(0, 2, 3, 1), mode=mode, padding_mode='border', align_corners=True)
        output = torch.cat((output0, output1, output2), dim=1)
        return output
