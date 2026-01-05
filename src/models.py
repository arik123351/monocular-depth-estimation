"""
Depth Estimation Models

This module contains various neural network architectures for monocular depth estimation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights


class DepthEncoder(nn.Module):
    """Encoder module using ResNet backbone."""
    
    def __init__(self, backbone='resnet50', pretrained=True):
        """
        Initialize the encoder.
        
        Args:
            backbone (str): ResNet backbone ('resnet18', 'resnet50', etc.)
            pretrained (bool): Use pretrained weights
        """
        super().__init__()
        
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Remove the final classification layer
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Returns:
            dict: Dictionary containing feature maps at different scales
        """
        features = {}
        
        x = self.layer0(x)
        features['layer0'] = x
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        return features


class DepthDecoder(nn.Module):
    """Decoder module to reconstruct depth map."""
    
    def __init__(self, num_channels=1, encoder_channels=None):
        """
        Initialize the decoder.
        
        Args:
            num_channels (int): Number of output channels (1 for single depth map)
            encoder_channels (dict): Dictionary of encoder output channels for each layer
                                    If None, assumes ResNet50 (default legacy behavior)
        """
        super().__init__()
        
        # Set default encoder channels for ResNet50 if not provided
        if encoder_channels is None:
            encoder_channels = {
                'layer0': 64,
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024,
                'layer4': 2048
            }
        
        # Extract channel dimensions
        c0 = encoder_channels.get('layer0', 64)
        c1 = encoder_channels.get('layer1', 256)
        c2 = encoder_channels.get('layer2', 512)
        c3 = encoder_channels.get('layer3', 1024)
        c4 = encoder_channels.get('layer4', 2048)
        
        # Decoder blocks with dynamic channel handling for skip connections
        # layer4: (B, c4, H/32, W/32)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # After upsample4 and concat with layer3
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512 + c3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # After upsample3 and concat with layer2
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + c2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # After upsample2 and concat with layer1
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128 + c1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # After upsample1 and concat with layer0
        self.decoder0 = nn.Sequential(
            nn.Conv2d(64 + c0, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.upsample0 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # Final layer to output depth
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        Forward pass through decoder.
        
        Args:
            features (dict): Feature maps from encoder
            
        Returns:
            torch.Tensor: Predicted depth map
        """
        # Encoder output dimensions:
        # layer0: stride=4  (B, 64, H/4, W/4)
        # layer1: stride=4  (B, 256, H/4, W/4)
        # layer2: stride=8  (B, 512, H/8, W/8)
        # layer3: stride=16 (B, 1024, H/16, W/16)
        # layer4: stride=32 (B, 2048, H/32, W/32)
        
        # Start from layer4
        x = self.decoder4(features['layer4'])  # (B, 512, H/32, W/32)
        x = self.upsample4(x)  # (B, 512, H/16, W/16)
        
        # Concatenate with layer3
        x = torch.cat([x, features['layer3']], dim=1)  # (B, 512+1024, H/16, W/16)
        x = self.decoder3(x)  # (B, 256, H/16, W/16)
        x = self.upsample3(x)  # (B, 256, H/8, W/8)
        
        # Concatenate with layer2
        x = torch.cat([x, features['layer2']], dim=1)  # (B, 256+512, H/8, W/8)
        x = self.decoder2(x)  # (B, 128, H/8, W/8)
        x = self.upsample2(x)  # (B, 128, H/4, W/4)
        
        # Concatenate with layer1
        x = torch.cat([x, features['layer1']], dim=1)  # (B, 128+256, H/4, W/4)
        x = self.decoder1(x)  # (B, 64, H/4, W/4)
        x = self.upsample1(x)  # (B, 64, H/2, W/2) - but layer0 is at H/4, W/4
        
        # Need to upsample x to match layer0 spatial dimensions (H/4, W/4)
        # Current: x is (B, 64, H/2, W/2), layer0 is (B, 64, H/4, W/4)
        # We need to downsample x back or not upsample as much
        
        # Better approach: upsample to H/4, W/4 then concatenate
        if x.shape[2] != features['layer0'].shape[2]:
            # Upsample x to match layer0 spatial dimensions
            x = torch.nn.functional.interpolate(
                x, size=(features['layer0'].shape[2], features['layer0'].shape[3]),
                mode='bilinear', align_corners=False
            )
        
        # Concatenate with layer0
        x = torch.cat([x, features['layer0']], dim=1)  # (B, 64+64, H/4, W/4)
        x = self.decoder0(x)  # (B, 32, H/4, W/4)
        x = self.upsample0(x)  # (B, 32, H, W) - 4x upsample to get back to original size
        
        # Final depth prediction
        x = self.final(x)  # (B, 1, H, W)
        
        return x


class DepthEstimationNet(nn.Module):
    """Complete depth estimation network."""
    
    def __init__(self, backbone='resnet50', pretrained=True):
        """
        Initialize the depth estimation network.
        
        Args:
            backbone (str): Encoder backbone architecture
            pretrained (bool): Use pretrained weights for encoder
        """
        super().__init__()
        self.encoder = DepthEncoder(backbone=backbone, pretrained=pretrained)
        
        # Get encoder channel dimensions based on backbone
        if backbone == 'resnet18':
            encoder_channels = {
                'layer0': 64,
                'layer1': 64,    # ResNet18: layer1 has 64 channels
                'layer2': 128,   # ResNet18: layer2 has 128 channels
                'layer3': 256,   # ResNet18: layer3 has 256 channels
                'layer4': 512    # ResNet18: layer4 has 512 channels
            }
        else:  # ResNet50 or default
            encoder_channels = {
                'layer0': 64,
                'layer1': 256,   # ResNet50: layer1 has 256 channels
                'layer2': 512,   # ResNet50: layer2 has 512 channels
                'layer3': 1024,  # ResNet50: layer3 has 1024 channels
                'layer4': 2048   # ResNet50: layer4 has 2048 channels
            }
        
        self.decoder = DepthDecoder(num_channels=1, encoder_channels=encoder_channels)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input RGB image (B, 3, H, W)
            
        Returns:
            torch.Tensor: Predicted depth map (B, 1, H, W)
        """
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


def create_model(model_type='depth', backbone='resnet50', pretrained=True, device='cuda'):
    """
    Create and return a model.
    
    Args:
        model_type (str): Type of model to create
        backbone (str): Encoder backbone
        pretrained (bool): Use pretrained weights
        device (str): Device to move model to
        
    Returns:
        nn.Module: The model
    """
    if model_type == 'depth':
        model = DepthEstimationNet(backbone=backbone, pretrained=pretrained)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    print("Testing Depth Estimation Model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(device=device)
    
    # Create dummy input
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device: {device}")
    print("Model test passed!")
