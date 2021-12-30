from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import ConvBnReLU, depth_regression
from .patchmatch import PatchMatch


class FeatureNet(nn.Module):
    """Feature Extraction Network: to extract features of original images from each view"""

    def __init__(self):
        """Initialize different layers in the network"""

        super(FeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage 1 to stage 3
                keys are 1, 2, and 3
        """
        output_feature: Dict[int, torch.Tensor] = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature[3] = self.output1(conv10)
        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7
        del conv10

        output_feature[2] = self.output2(intra_feat)
        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv4

        output_feature[1] = self.output3(intra_feat)
        del intra_feat

        return output_feature


class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        """Initialize"""

        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(
        self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method

        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H//2, W//2)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )

        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (depth_max - depth_min).view(batch_size, 1, 1, 1)

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        # depth residual
        res = self.res(self.conv3(torch.cat((deconv, conv0), dim=1)))
        del conv0
        del deconv

        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res
        # convert the normalized depth back
        return depth * (depth_max - depth_min).view(batch_size, 1, 1, 1) + depth_min.view(batch_size, 1, 1, 1)


class PatchmatchNet(nn.Module):
    """ Implementation of complete structure of PatchmatchNet"""

    def __init__(
        self,
        patchmatch_interval_scale: List[float],
        propagation_range: List[int],
        patchmatch_iteration: List[int],
        patchmatch_num_sample: List[int],
        propagate_neighbors: List[int],
        evaluate_neighbors: List[int],
    ) -> None:
        """Initialize modules in PatchmatchNet

        Args:
            patchmatch_interval_scale: depth interval scale in patchmatch module
            propagation_range: propagation range
            patchmatch_iteration: patchmatch iteration number
            patchmatch_num_sample: patchmatch number of samples
            propagate_neighbors: number of propagation neighbors
            evaluate_neighbors: number of propagation neighbors for evaluation
        """
        super(PatchmatchNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [16, 32, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        for i in range(self.stages - 1):
            patchmatch = PatchMatch(
                propagation_out_range=propagation_range[i],
                patchmatch_iteration=patchmatch_iteration[i],
                patchmatch_num_sample=patchmatch_num_sample[i],
                patchmatch_interval_scale=patchmatch_interval_scale[i],
                num_feature=num_features[i],
                G=self.G[i],
                propagate_neighbors=self.propagate_neighbors[i],
                evaluate_neighbors=evaluate_neighbors[i],
                stage=i + 1,
            )
            setattr(self, f"patchmatch_{i+1}", patchmatch)

        self.upsample_net = Refinement()

    def forward(
        self,
        images: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[torch.Tensor]]]:
        """Forward method for PatchMatchNet

        Args:
            images: N images (B, 3, H, W) stored in list
            intrinsics: intrinsic 3x3 matrices for all images (B, N, 3, 3)
            extrinsics: extrinsic 4x4 matrices for all images (B, N, 4, 4)
            depth_min: minimum virtual depth (B, 1)
            depth_max: maximum virtual depth (B, 1)

        Returns:
            output tuple of PatchMatchNet, containing refined depthmap, depth patchmatch, and photometric confidence.
        """
        assert len(images) == intrinsics.size()[1], "Different number of images and intrinsic matrices"
        assert len(images) == extrinsics.size()[1], 'Different number of images and extrinsic matrices'
        images, intrinsics, orig_height, orig_width = adjust_image_dims(images, intrinsics)
        ref_image = images[0]
        _, _, ref_height, ref_width = ref_image.size()

        # step 1. Multi-scale feature extraction
        features: List[Dict[int, torch.Tensor]] = []
        for img in images:
            output_feature = self.feature(img)
            features.append(output_feature)
        del images
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        device = intrinsics.device
        depth = torch.empty(0, device=device)
        depths: List[torch.Tensor] = []
        score = torch.empty(0, device=device)
        view_weights = torch.empty(0, device=device)
        depth_patchmatch: Dict[int, List[torch.Tensor]] = {}

        scale = 0.125
        for stage in range(self.stages - 1, 0, -1):
            src_features_l = [src_fea[stage] for src_fea in src_features]

            # Create projection matrix for specific stage
            intrinsics_l = intrinsics.clone()
            intrinsics_l[:, :, :2] *= scale
            proj = extrinsics.clone()
            proj[:, :, :3, :4] = torch.matmul(intrinsics_l, extrinsics[:, :, :3, :4])
            proj_l = torch.unbind(proj, 1)
            ref_proj, src_proj = proj_l[0], proj_l[1:]
            scale *= 2.0

            # Need conditional since TorchScript only allows "getattr" access with string literals
            if stage == 3:
                depths, score, view_weights = self.patchmatch_3(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 2:
                depths, score, view_weights = self.patchmatch_2(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 1:
                depths, score, view_weights = self.patchmatch_1(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )

            depth_patchmatch[stage] = depths
            depth = depths[-1].detach()

            if stage > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth, scale_factor=2.0, mode="nearest")
                view_weights = F.interpolate(view_weights, scale_factor=2.0, mode="nearest")

        del ref_feature
        del src_features

        # step 3. Refinement
        depth = self.upsample_net(ref_image, depth, depth_min, depth_max)
        if ref_width != orig_width or ref_height != orig_height:
            depth = F.interpolate(depth, size=[orig_height, orig_width], mode='bilinear', align_corners=False)
        depth_patchmatch[0] = [depth]

        if self.training:
            return depth, torch.empty(0, device=device), depth_patchmatch
        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(
                score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)
            ).long().clamp(0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(
                photometric_confidence, size=[orig_height, orig_width], mode="nearest").squeeze(1)

            return depth, photometric_confidence, depth_patchmatch


def adjust_image_dims(
        images: List[torch.Tensor], intrinsics: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    # stretch or compress image slightly to ensure width and height are multiples of 8
    _, _, ref_height, ref_width = images[0].size()
    for i in range(len(images)):
        _, _, height, width = images[i].size()
        new_height = int(round(height / 8)) * 8
        new_width = int(round(width / 8)) * 8
        if new_width != width or new_height != height:
            intrinsics[:, i, 0] *= new_width / width
            intrinsics[:, i, 1] *= new_height / height
            images[i] = nn.functional.interpolate(
                images[i], size=[new_height, new_width], mode='bilinear', align_corners=False)

    return images, intrinsics, ref_height, ref_width


def patchmatchnet_loss(
    depth_patchmatch: Dict[int, List[torch.Tensor]],
    depth_gt: List[torch.Tensor],
    mask: List[torch.Tensor],
) -> torch.Tensor:
    """Patchmatch Net loss function

    Args:
        depth_patchmatch: depth map predicted by patchmatch net
        depth_gt: ground truth depth map
        mask: mask for filter valid points

    Returns:
        loss: result loss value
    """
    loss = 0
    for i in range(0, 4):
        gt_depth = depth_gt[i][mask[i]]
        for depth in depth_patchmatch[i]:
            loss = loss + F.smooth_l1_loss(depth[mask[i]], gt_depth, reduction="mean")

    return loss
