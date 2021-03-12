import torch
import torch.nn as nn
import torch.nn.functional

from torch import Tensor


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1,
                 dilation: int = 1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1,
                 dilation: int = 1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1,
                 dilation: int = 1):
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))


def differentiable_warping(src_fea: Tensor, src_proj: Tensor, ref_proj: Tensor, depth_samples: Tensor) -> Tensor:
    # src_fea: [B, C, H_in, W_in]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, num_depth, H_out, W_out]
    # out: [B, C, num_depth, H_out, W_out]
    batch, num_depth, height, width = depth_samples.shape
    channels = src_fea.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x, dtype=torch.float32))).unsqueeze(0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_depth, 1)  # [B, 3, H*W]

        xyz = xyz * depth_samples.view(batch, 1, num_depth, height * width) + trans.view(batch, 3, 1, 1)  # [B, 3, num_depth, H*W]

        # avoid negative depth
        negative_depth_mask = xyz[:, 2:] <= 1e-3
        xyz[:, 0:1][negative_depth_mask] = float(width)
        xyz[:, 1:2][negative_depth_mask] = float(height)
        xyz[:, 2:3][negative_depth_mask] = 1.0
        proj_xy = xyz[:, :2, :, :] / xyz[:, 2:3, :, :]  # [B, 2, num_depth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, num_depth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, num_depth, H*W, 2]

    return nn.functional.grid_sample(src_fea, proj_xy.view(batch, num_depth * height, width, 2), mode='bilinear',
                                     padding_mode='zeros', align_corners=False).view(batch, channels, num_depth, height, width)


def is_empty(x: Tensor) -> bool:
    return x.numel() == 0
