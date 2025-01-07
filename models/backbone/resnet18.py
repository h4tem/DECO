import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # load the base model
        resnet = resnet18(pretrained=pretrained)
        # remove the classifier head
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        # returns feature map shape: (B, C, H, W)
        return self.body(x)
