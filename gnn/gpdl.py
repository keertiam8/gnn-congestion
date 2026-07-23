"""
GPDL: CircuitNet's official congestion-prediction baseline model.

Architecture reconstructed to match the state_dict shapes in
checkpoints/congestion.pth exactly (verified layer-by-layer against
CircuitNet's routability_ir_drop_prediction/models/gpdl.py). This is a
U-Net-style CNN (conv/pool encoder, deconv/skip-concat decoder) operating on
dense (C, H, W) grid feature maps -- NOT the graph-based CongestionGNN in
model.py. The two architectures are incompatible; this module exists to load
CircuitNet's pretrained weights as a baseline predictor to benchmark the
closed-loop GNN approach against, not to warm-start CongestionGNN.

checkpoint: in_channels=3 (macro_region, RUDY, RUDY_pin), out_channels=1
(congestion_label).
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32):
        super().__init__()
        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Sequential(
            nn.Conv2d(64, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
        )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.pool1(h1)
        h3 = self.c2(h2)
        h4 = self.pool2(h3)
        h5 = self.c3(h4)
        return h5, h2  # bottleneck features + skip connection


class Decoder(nn.Module):
    def __init__(self, out_dim=1, in_dim=32):
        super().__init__()
        self.conv1 = conv(in_dim, 32)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(32 + 16, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, out_dim, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d2 = self.upc1(d1)
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        return self.conv3(d4)


class GPDL(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()
        self.encoder = Encoder(in_dim=in_channels)
        self.decoder = Decoder(out_dim=out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_pretrained_gpdl(checkpoint_path, in_channels=3, out_channels=1, device="cpu"):
    """Load CircuitNet's pretrained GPDL weights (checkpoints/congestion.pth)."""
    model = GPDL(in_channels=in_channels, out_channels=out_channels)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model.load_state_dict(OrderedDict(state_dict), strict=True)
    return model.to(device).eval()
