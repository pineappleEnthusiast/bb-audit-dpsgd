"""WideResNet implementation from https://github.com/OsvaldFrisk/dp-not-all-noise-is-equal/blob/master/src/networks.py
Paper: https://arxiv.org/pdf/2110.06255"""
"""WideResNet 16-4 with Weight Standardization and GroupNorm.
    This implementation uses Weight Standardization and GroupNorm by default.
    """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WSConv2d(nn.Conv2d):
    """2D Convolution with Weight Standardization and affine gain+bias."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        # Initialize gain as ones
        self.gain = nn.Parameter(torch.ones(out_channels))
        # Initialize bias as zeros
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def standardize_weight(self, eps=1e-4):
        """Apply weight standardization with affine gain."""
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True)
        fan_in = torch.prod(torch.tensor(weight.shape[:-1]).float())
        gain = self.gain.view(-1, 1, 1, 1)
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * gain
        shift = mean * scale
        return (weight * scale - shift)

    def forward(self, x):
        weight = self.standardize_weight()
        x = F.conv2d(x, weight, None, self.stride,
                     self.padding, self.dilation, self.groups)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x

class StochDepth(nn.Module):
    """Batchwise Dropout (Stochastic Depth)."""
    def __init__(self, drop_rate: float, scale_by_keep: bool = False):
        super().__init__()
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x
        batch_size = x.size(0)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = torch.floor(keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device))
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excite module."""
    def __init__(self, in_ch: int, out_ch: int, se_ratio: float = 0.5,
                 hidden_ch: int = None, activation: str = 'relu'):
        super().__init__()
        if se_ratio is None:
            if hidden_ch is None:
                raise ValueError('Must provide one of se_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(in_ch * se_ratio))
        
        self.activation = getattr(F, activation)
        self.fc0 = nn.Linear(in_ch, self.hidden_ch)
        self.fc1 = nn.Linear(self.hidden_ch, out_ch)

    def forward(self, x):
        # Average over HW dimensions
        h = x.mean(dim=[2, 3])
        h = self.activation(self.fc0(h))
        h = self.fc1(h)
        # Broadcast along H, W dimensions
        h = torch.sigmoid(h)[:, :, None, None]
        return x * h




        
        self.num_output_classes = num_classes
        self.width = width
        self.use_skip_init = use_skip_init
        self.use_skip_paths = use_skip_paths
        self.dropout_rate = dropout_rate
        self.resnet_blocks = (depth - 4) // 6
        
        # Use Weight Standardization and GroupNorm by default
        self.conv_fn = WSConv2d
        self.norm_fn = functools.partial(nn.GroupNorm, groups=16)
        
        # Activation selection with scaling
        activations = {
            # Non-scaled activations
            'identity': lambda x: x,
            'celu': nn.CELU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
            'glu': lambda x: nn.GLU(dim=1)(x),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'log_sigmoid': nn.LogSigmoid(),
            'log_softmax': nn.LogSoftmax(dim=1),
            'relu': nn.ReLU(inplace=True),
            'relu6': nn.ReLU6(inplace=True),
            'selu': nn.SELU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU(inplace=True),
            'swish': nn.SiLU(inplace=True),
            'soft_sign': nn.Softsign(),
            'softplus': nn.Softplus(),
            'tanh': nn.Tanh(),
            
            # Scaled activations
            'scaled_celu': lambda x: 1.270926833152771 * nn.CELU(inplace=True)(x),
            'scaled_elu': lambda x: 1.2716004848480225 * nn.ELU(inplace=True)(x),
            'scaled_gelu': lambda x: 1.7015043497085571 * nn.GELU()(x),
            'scaled_glu': lambda x: 1.8484294414520264 * nn.GLU(dim=1)(x),
            'scaled_leaky_relu': lambda x: 1.70590341091156 * nn.LeakyReLU(0.1, inplace=True)(x),
            'scaled_log_sigmoid': lambda x: 1.9193484783172607 * nn.LogSigmoid()(x),
            'scaled_log_softmax': lambda x: 1.0002083778381348 * nn.LogSoftmax(dim=1)(x),
            'scaled_relu': lambda x: 1.7139588594436646 * nn.ReLU(inplace=True)(x),
            'scaled_relu6': lambda x: 1.7131484746932983 * nn.ReLU6(inplace=True)(x),
            'scaled_selu': lambda x: 1.0008515119552612 * nn.SELU(inplace=True)(x),
            'scaled_sigmoid': lambda x: 4.803835391998291 * nn.Sigmoid()(x),
            'scaled_silu': lambda x: 1.7881293296813965 * nn.SiLU(inplace=True)(x),
            'scaled_swish': lambda x: 1.7881293296813965 * nn.SiLU(inplace=True)(x),
            'scaled_soft_sign': lambda x: 2.338853120803833 * nn.Softsign()(x),
            'scaled_softplus': lambda x: 1.9203323125839233 * nn.Softplus()(x),
            'scaled_tanh': lambda x: 1.5939117670059204 * nn.Tanh()(x)
        }
        self.activation = activations[activation]
        
        # First convolution
        self.conv1 = self.conv_fn(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Residual blocks
        self.block1 = self._make_layer(16 * width, (1, 1), 'Block_1')
        self.block2 = self._make_layer(32 * width, (2, 2), 'Block_2')
        self.block3 = self._make_layer(64 * width, (2, 2), 'Block_3')
        
        # Final layers
        self.final_norm = self.norm_fn(64 * width)
        self.fc = nn.Linear(64 * width, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, WSConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, width, strides, name):
        layers = []
        for i in range(self.resnet_blocks):
            if self.use_skip_paths:
                # Skip branch
                skip = nn.Sequential(
                    self.activation,
                    self.norm_fn(width),
                    self.conv_fn(width, width, kernel_size=1, stride=strides if i == 0 else (1, 1), bias=False)
                )
            else:
                skip = None
            
            # Residual branch
            residual = nn.Sequential(
                self.activation,
                self.norm_fn(width),
                self.conv_fn(width, width, kernel_size=3, stride=strides if i == 0 else (1, 1), padding=1, bias=False),
                self.activation,
                self.norm_fn(width),
                self.conv_fn(width, width, kernel_size=3, stride=1, padding=1, bias=False)
            )
            
            # Skip initialization
            if self.use_skip_init:
                skip_scale = nn.Parameter(torch.zeros(1))
                residual_scale = nn.Parameter(torch.zeros(1))
                residual = nn.Sequential(residual, skip_scale)
            
            layers.append(nn.Sequential(
                SkipPath(skip) if skip else nn.Identity(),
                residual,
                AddPath(skip_scale if self.use_skip_init else None)
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.activation(x)
        x = self.final_norm(x)
        x = x.mean(dim=[2, 3])
        
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.fc(x)
        return x

class SkipPath(nn.Module):
    def __init__(self, skip):
        super().__init__()
        self.skip = skip
    
    def forward(self, x):
        return self.skip(x)

class AddPath(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, skip):
        if self.scale is not None:
            x = x * self.scale
        return x + skip

class BasicBlock(nn.Module):
    """Basic residual block with GroupNorm and Weight Standardization."""
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.GroupNorm(16, in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = WSConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.norm2 = nn.GroupNorm(16, out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = WSConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and WSConv2d(in_planes, out_planes, 
            kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.norm1(x))
        else:
            out = self.relu1(self.norm1(x))
        out = self.relu2(self.norm2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """Network block with GroupNorm and Weight Standardization."""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, 
                               i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    """WideResNet 16-4 with GroupNorm (group_size=16) and Weight Standardization.
    
    Args:
        input_shape: Shape of the input tensor (C, H, W)
        out_dim: Number of output classes
        width: Width factor for the network (default: 4)
        dropout_rate: Dropout rate (default: 0.0)
    """
    def __init__(self, input_shape, out_dim=10, width=4, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*width, 32*width, 64*width]
        self.depth = 16  # Fixed depth for WideResNet 16-4
        assert((self.depth-4)%6 == 0)
        n = int((self.depth-4)/6)
        block = BasicBlock
        
        # 1st conv before any network block
        self.conv1 = WSConv2d(input_shape[1], nChannels[0], kernel_size=3, stride=1,
                              padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.norm1 = nn.GroupNorm(16, nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], out_dim)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.norm1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
