import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# Define the AffineNet model using pre-trained ResNet-18
class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()
        # Load pre-trained ResNet-101 model (without final fc layer)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
        # Freeze the pre-trained layers 
        for param in self.resnet_features.parameters():
            param.requires_grad = False
            
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * 8, 512*4)  # Adjust input size based on ResNet output
        self.fc2 = nn.Linear(512*2, 6)  # Output: 6 parameters for affine transformation matrix
        
        

    def forward(self, x):
        fixed_features = self.resnet_features(x[0])
        # fixed_features = self.avgpool(fixed_features)
        # print(f"fixed_features size: {fixed_features.shape}")
        moving_features = self.resnet_features(x[1])
        # moving_features = self.avgpool(moving_features)
        # print(f"moving_features size: {moving_features.shape}")
        # Flatten the pooled features
        fixed_features = fixed_features.view(fixed_features.size(0), -1)
        moving_features = moving_features.view(moving_features.size(0), -1)
        # Concatenate features from fixed and moving images
        x = torch.cat([fixed_features, moving_features], dim=1)
        # x = self.fc1(x)
        x = self.fc2(x)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp = torch.tensor([1.0,0,0,0,1.0,0]).to(device)
        
        x = temp + 0.1*x
        
        return x
