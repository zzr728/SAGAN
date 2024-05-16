import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
    

class ResidualBlock1(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock1, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.sc = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        s_c = self.main(x)
        y = self.sc(s_c)
        return s_c + y
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6,device = 'cuda:6',num_memory = 4):
        super(Generator, self).__init__()
        self.num_memory = num_memory

        self.layer1 = nn.Sequential(
            nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # Down-sampling layers.
        curr_dim = conv_dim
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2
        self.p_bn1 = self.position_bn(curr_dim=curr_dim,num_memory=self.num_memory)


        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2
        self.p_bn2 = self.position_bn(curr_dim=curr_dim,num_memory=self.num_memory)

        self.enc_layer3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim * 2
        
        # Bottleneck layers.

        layers = []
        layers.append(nn.Conv2d(curr_dim + int(np.log2(num_memory)+1), curr_dim, 1, 1, 0))
        layers.append(nn.InstanceNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))

        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.bn2 = nn.Sequential(*layers)


        # Up-sampling layers.
        self.gate1 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn1 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(curr_dim//2, curr_dim//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//4, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )


        self.gate2 = AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn2 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.gate3= AttentionGate(F_g=curr_dim,F_l=curr_dim,n_coefficients=curr_dim//2)
        self.dec_bn3 = ResidualBlock1(dim_in=2*curr_dim,dim_out=curr_dim)
        self.dec_layer3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        curr_dim = curr_dim // 2
        self.bn3 = ResidualBlock1(dim_in=3*curr_dim,dim_out=curr_dim)

        self.conv = nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.binary_encoding = binarize(torch.arange(0,self.num_memory,dtype=torch.int), int(np.log2(num_memory)+1))

    def forward(self, x,bs=1):
        y = self.layer1(x)

        y1 = self.enc_layer1(y)
        _,C,W,H = y1.shape
        y1 = y1.view(bs,self.num_memory,C,W,H)
        y1 = self.add_condition(y1,bs,num_windows=self.num_memory)
        y1 = y1.view(bs * self.num_memory,-1,W,H)
        y1 = self.p_bn1(y1)
        sc = []
        sc.append(y1)

        y2 = self.enc_layer2(y1)
        _,C,W,H = y2.shape
        y2 = y2.view(bs,self.num_memory,C,W,H)
        y2 = self.add_condition(y2,bs,num_windows=self.num_memory)
        y2 = y2.view(bs * self.num_memory,-1,W,H)
        y2 = self.p_bn2(y2)
        sc.append(y2)

        y3 = self.enc_layer3(y2)
        sc.append(y3)

        y = y3
        B,C,W,H = y.shape
        y = y.view(bs,self.num_memory,C,W,H)
        y_p = self.add_condition(y,bs,num_windows=self.num_memory)
        y_p = y_p.view(bs * self.num_memory,-1,W,H)
        embedding = self.bn2(y_p)

        i = 0
        sc1 = self.gate1(gate=embedding,skip_connection=embedding)
        out1 = torch.concat([embedding,sc1],dim=1)
        out1 = self.dec_bn1(out1)
        out1 = self.dec_layer1(out1)

        i += 1
        sc2 = self.gate2(gate=out1,skip_connection=sc[-1-i])
        out2 = torch.concat([out1,sc2],dim=1)
        out2 = self.dec_bn2(out2)
        out2 = self.dec_layer2(out2)

        i += 1
        sc3 = self.gate3(gate=out2,skip_connection=sc[-1-i])
        out3 = torch.concat([out2,sc3],dim=1)
        out3 = self.dec_bn3(out3)
        out3 = self.dec_layer3(out3)

        out1 = self.up_conv1(out1)
        out2 = self.up_conv2(out2)
        out = torch.concat([out1,out2,out3],dim=1)
        out = self.bn3(out)

        output = self.conv(out)

        if self.training:
            # spatial binary mask
            mask = torch.ones(output.size(0), 1, output.size(-2), output.size(-1)).to(output.device) * 0.95
            mask = torch.bernoulli(mask).float()
            output = mask * output + (1. - mask) * x
        fake1 = torch.tanh(output+x)
    
        return fake1
    
    def add_condition(self, x, bs, num_windows):
        condition = self.binary_encoding.to(x.device).view(1, self.binary_encoding.shape[0], self.binary_encoding.shape[1], 1, 1)
        condition = condition.expand(bs, -1, -1, x.size(-2), x.size(-1)).contiguous().float()
        x = torch.cat((x, condition), dim=2)
        return x
    
    def position_bn(self,curr_dim,num_memory = 4):
        layers = []
        layers.append(nn.Conv2d(curr_dim + int(np.log2(num_memory)+1), curr_dim, 1, 1, 0))
        layers.append(nn.InstanceNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))
        bn = nn.Sequential(*layers)
        return bn

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return h, out_src

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def binarize(integer, num_bits=8):   
    """Turn integer tensor to binary representation.        
    Args:           
    integer : torch.Tensor, tensor with integers           
    num_bits : Number of bits to specify the precision. Default: 8.       
    Returns:           
    Tensor: Binary tensor. Adds last dimension to original tensor for           
    bits.    
    """   
    dtype = integer.type()   
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)   
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))   
    out = integer.unsqueeze(-1) / 2 ** exponent_bits   
    return (out - (out % 1)) % 2

class AttentionGate(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionGate, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi

        return out
    

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x