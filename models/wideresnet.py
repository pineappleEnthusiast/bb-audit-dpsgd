"""WideResNet implementation from https://github.com/OsvaldFrisk/dp-not-all-noise-is-equal/blob/master/src/networks.py
Paper: https://arxiv.org/pdf/2110.06255"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WideResNet(nn.Module):
    """
    Simple ConvNet with WideResNet-like channel progression
    NO residual connections, NO normalization - just like your working ConvNet
    """
    def __init__(self, num_classes=10, widen_factor=4, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        
        self.embeddings = None
        # WideResNet channel progression: [16, 64, 128, 256] for widen_factor=4
        channels = [16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        
        # Feature layers - similar to your ConvNet but with WideResNet channels
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            
            # Second block
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            
            # Third block
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        
        # Classifier - similar to your ConvNet
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(channels[2] * 4 * 4, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes)
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
