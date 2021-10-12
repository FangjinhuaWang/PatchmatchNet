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
        patchmatch_interval_scale: List[float] = [0.005, 0.0125, 0.025],
        propagation_range: List[int] = [6, 4, 2],
        patchmatch_iteration: List[int] = [1, 2, 2],
        patchmatch_num_sample: List[int] = [8, 8, 16],
        propagate_neighbors: List[int] = [0, 8, 16],
        evaluate_neighbors: List[int] = [9, 9, 9],
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
        images: Dict[str, torch.Tensor],
        proj_matrices: Dict[str, torch.Tensor],
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[torch.Tensor]]]:
        """Forward method for PatchMatchNet

        Args:
            images: different stages of images (B, 3, H, W) stored in the dictionary
            proj_matrices: different stages of camera projection matrices (B, 4, 4) stored in the dictionary
            depth_min: minimum virtual depth (B, )
            depth_max: maximum virtual depth (B, )

        Returns:
            output tuple of PatchMatchNet, containing refined depthmap, depth patchmatch, and photometric confidence.
        """
        imgs_0 = torch.unbind(images["stage_0"], 1)
        del images

        ref_image = imgs_0[0]

        proj_mtx = {
            0: torch.unbind(proj_matrices["stage_0"].float(), 1),
            1: torch.unbind(proj_matrices["stage_1"].float(), 1),
            2: torch.unbind(proj_matrices["stage_2"].float(), 1),
            3: torch.unbind(proj_matrices["stage_3"].float(), 1)
        }
        del proj_matrices

        assert len(imgs_0) == len(proj_mtx[0]), "Different number of images and projection matrices"

        # step 1. Multi-scale feature extraction
        features: List[Dict[int, torch.Tensor]] = []
        for img in imgs_0:
            output_feature = self.feature(img)
            features.append(output_feature)
        del imgs_0
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = torch.empty(0)
        depths: List[torch.Tensor] = []
        score = torch.empty(0)
        view_weights = torch.empty(0)
        depth_patchmatch: Dict[int, List[torch.Tensor]] = {}

        for stage in range(self.stages - 1, 0, -1):
            src_features_l = [src_fea[stage] for src_fea in src_features]
            ref_proj, src_projs = proj_mtx[stage][0], proj_mtx[stage][1:]
            # Need conditional since TorchScript only allows "getattr" access with string literals
            if stage == 3:
                depths, _, view_weights = self.patchmatch_3(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 2:
                depths, _, view_weights = self.patchmatch_2(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 1:
                depths, score, _ = self.patchmatch_1(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_projs,
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

        if self.training:
            return depth, torch.empty(0), depth_patchmatch
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
            photometric_confidence = F.interpolate(photometric_confidence, scale_factor=2.0, mode="nearest").squeeze(1)

            return depth, photometric_confidence, depth_patchmatch


def patchmatchnet_loss(
    depth_patchmatch: Dict[int, List[torch.Tensor]],
    depth_gt: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
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
        mask_i = mask[f"stage_{i}"] > 0.5
        gt_depth = depth_gt[f"stage_{i}"][mask_i]
        for depth in depth_patchmatch[i]:
            loss = loss + F.smooth_l1_loss(depth[mask_i], gt_depth, reduction="mean")

    return loss
