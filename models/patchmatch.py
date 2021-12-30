"""
PatchmatchNet uses the following main steps:

1. Initialization: generate random hypotheses;
2. Propagation: propagate hypotheses to neighbors;
3. Evaluation: compute the matching costs for all the hypotheses and choose best solutions.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ConvBnReLU3D, differentiable_warping, is_empty


class DepthInitialization(nn.Module):
    """Initialization Stage Class"""

    def __init__(self, patchmatch_num_sample: int = 1) -> None:
        """Initialize method

        Args:
            patchmatch_num_sample: number of samples used in patchmatch process
        """
        super(DepthInitialization, self).__init__()
        self.patchmatch_num_sample = patchmatch_num_sample

    def forward(
        self,
        min_depth: torch.Tensor,
        max_depth: torch.Tensor,
        height: int,
        width: int,
        depth_interval_scale: float,
        device: torch.device,
        depth: torch.Tensor
    ) -> torch.Tensor:
        """Forward function for depth initialization

        Args:
            min_depth: minimum virtual depth, (B, )
            max_depth: maximum virtual depth, (B, )
            height: height of depth map
            width: width of depth map
            depth_interval_scale: depth interval scale
            device: device on which to place tensor
            depth: current depth (B, 1, H, W)

        Returns:
            depth_sample: initialized sample depth map by randomization or local perturbation (B, Ndepth, H, W)
        """
        batch_size = min_depth.size()[0]
        inverse_min_depth = 1.0 / min_depth
        inverse_max_depth = 1.0 / max_depth
        if is_empty(depth):
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            patchmatch_num_sample = 48
            # [B,Ndepth,H,W]
            depth_sample = torch.rand(
                size=(batch_size, patchmatch_num_sample, height, width), device=device
            ) + torch.arange(start=0, end=patchmatch_num_sample, step=1, device=device).view(
                1, patchmatch_num_sample, 1, 1
            )

            depth_sample = inverse_max_depth.view(batch_size, 1, 1, 1) + depth_sample / patchmatch_num_sample * (
                inverse_min_depth.view(batch_size, 1, 1, 1) - inverse_max_depth.view(batch_size, 1, 1, 1)
            )

            return 1.0 / depth_sample

        elif self.patchmatch_num_sample == 1:
            return depth.detach()
        else:
            # other Patchmatch, local perturbation is performed based on previous result
            # uniform samples in an inversed depth range
            depth_sample = (
                torch.arange(-self.patchmatch_num_sample // 2, self.patchmatch_num_sample // 2, 1, device=device)
                .view(1, self.patchmatch_num_sample, 1, 1).repeat(batch_size, 1, height, width).float()
            )
            inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_interval_scale
            inverse_depth_interval = inverse_depth_interval.view(batch_size, 1, 1, 1)

            depth_sample = 1.0 / depth.detach() + inverse_depth_interval * depth_sample

            depth_clamped = []
            del depth
            for k in range(batch_size):
                depth_clamped.append(
                    torch.clamp(depth_sample[k], min=inverse_max_depth[k], max=inverse_min_depth[k]).unsqueeze(0)
                )

            return 1.0 / torch.cat(depth_clamped, dim=0)


class Propagation(nn.Module):
    """ Propagation module implementation"""

    def __init__(self) -> None:
        """Initialize method"""
        super(Propagation, self).__init__()

    def forward(self, depth_sample: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # [B,D,H,W]
        """Forward method of adaptive propagation

        Args:
            depth_sample: sample depth map, in shape of [batch, num_depth, height, width],
            grid: 2D grid for bilinear gridding, in shape of [batch, neighbors*H, W, 2]

        Returns:
            propagate depth: sorted propagate depth map [batch, num_depth+num_neighbors, height, width]
        """
        batch, num_depth, height, width = depth_sample.size()
        num_neighbors = grid.size()[1] // height
        propagate_depth_sample = F.grid_sample(
            depth_sample[:, num_depth // 2, :, :].unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        ).view(batch, num_neighbors, height, width)
        return torch.sort(torch.cat((depth_sample, propagate_depth_sample), dim=1), dim=1)[0]


class Evaluation(nn.Module):
    """Evaluation module for adaptive evaluation step in Learning-based Patchmatch
    Used to compute the matching costs for all the hypotheses and choose best solutions.
    """

    def __init__(self, G: int = 8) -> None:
        """Initialize method`

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(Evaluation, self).__init__()

        self.G = G
        self.pixel_wise_net = PixelwiseNet(self.G)
        self.softmax = nn.LogSoftmax(dim=1)
        self.similarity_net = SimilarityNet(self.G)

    def forward(
        self,
        ref_feature: torch.Tensor,
        src_features: List[torch.Tensor],
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_sample: torch.Tensor,
        grid: torch.Tensor,
        weight: torch.Tensor,
        view_weights: torch.Tensor,
        is_inverse: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method for adaptive evaluation

        Args:
            ref_feature: feature from reference view, (B, C, H, W)
            src_features: features from (Nview-1) source views, (Nview-1) * (B, C, H, W), where Nview is the number of
                input images (or views) of PatchmatchNet
            ref_proj: projection matrix of reference view, (B, 4, 4)
            src_projs: source matrices of source views, (Nview-1) * (B, 4, 4), where Nview is the number of input
                images (or views) of PatchmatchNet
            depth_sample: sample depth map, (B,Ndepth,H,W)
            grid: grid, (B, evaluate_neighbors*H, W, 2)
            weight: weight, (B,Ndepth,1,H,W)
            view_weights: Tensor to store weights of source views, in shape of (B,Nview-1,H,W),
                Nview-1 represents the number of source views
            is_inverse: Flag for inverse depth regression

        Returns:
            depth_sample: expectation of depth sample, (B,H,W)
            score: probability map, (B,Ndepth,H,W)
            view_weights: optional, Tensor to store weights of source views, in shape of (B,Nview-1,H,W),
                Nview-1 represents the number of source views
        """
        batch, feature_channel, height, width = ref_feature.size()
        device = ref_feature.device

        num_depth = depth_sample.size()[1]
        assert (
            len(src_features) == len(src_projs)
        ), "Patchmatch Evaluation: Different number of images and projection matrices"
        if not is_empty(view_weights):
            assert (
                len(src_features) == view_weights.size()[1]
            ), "Patchmatch Evaluation: Different number of images and view weights"

        # Change to a tensor with value 1e-5
        pixel_wise_weight_sum = 1e-5 * torch.ones((batch, 1, 1, height, width), dtype=torch.float32, device=device)
        ref_feature = ref_feature.view(batch, self.G, feature_channel // self.G, 1, height, width)
        similarity_sum = torch.zeros((batch, self.G, num_depth, height, width), dtype=torch.float32, device=device)

        i = 0
        view_weights_list = []
        for src_feature, src_proj in zip(src_features, src_projs):
            warped_feature = differentiable_warping(
                src_feature, src_proj, ref_proj, depth_sample
            ).view(batch, self.G, feature_channel // self.G, num_depth, height, width)
            # group-wise correlation
            similarity = (warped_feature * ref_feature).mean(2)
            # pixel-wise view weight
            if is_empty(view_weights):
                view_weight = self.pixel_wise_net(similarity)
                view_weights_list.append(view_weight)
            else:
                # reuse the pixel-wise view weight from first iteration of Patchmatch on stage 3
                view_weight = view_weights[:, i].unsqueeze(1)  # [B,1,H,W]
                i = i + 1

            similarity_sum += similarity * view_weight.unsqueeze(1)
            pixel_wise_weight_sum += view_weight.unsqueeze(1)

        # aggregated matching cost across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum)  # [B, G, Ndepth, H, W]
        # adaptive spatial cost aggregation
        score = self.similarity_net(similarity, grid, weight)  # [B, G, Ndepth, H, W]
        # apply softmax to get probability
        score = torch.exp(self.softmax(score))

        if is_empty(view_weights):
            view_weights = torch.cat(view_weights_list, dim=1)  # [B,4,H,W], 4 is the number of source views

        if is_inverse:
            # depth regression: inverse depth regression
            depth_index = torch.arange(0, num_depth, 1, device=device).view(1, num_depth, 1, 1)
            depth_index = torch.sum(depth_index * score, dim=1)

            inverse_min_depth = 1.0 / depth_sample[:, -1, :, :]
            inverse_max_depth = 1.0 / depth_sample[:, 0, :, :]
            depth_sample = inverse_max_depth + depth_index / (num_depth - 1) * (inverse_min_depth - inverse_max_depth)
            depth_sample = 1.0 / depth_sample
        else:
            # depth regression: expectation
            depth_sample = torch.sum(depth_sample * score, dim=1)

        return depth_sample, score, view_weights.detach()


class PatchMatch(nn.Module):
    """Patchmatch module"""

    def __init__(
        self,
        propagation_out_range: int = 2,
        patchmatch_iteration: int = 2,
        patchmatch_num_sample: int = 16,
        patchmatch_interval_scale: float = 0.025,
        num_feature: int = 64,
        G: int = 8,
        propagate_neighbors: int = 16,
        evaluate_neighbors: int = 9,
        stage: int = 3,
    ) -> None:
        """Initialize method

        Args:
            propagation_out_range: range of propagation out,
            patchmatch_iteration: number of iterations in patchmatch,
            patchmatch_num_sample: number of samples in patchmatch,
            patchmatch_interval_scale: interval scale,
            num_feature: number of features,
            G: the feature channels of input will be divided evenly into G groups,
            propagate_neighbors: number of neighbors to be sampled in propagation,
            stage: number of stage,
            evaluate_neighbors: number of neighbors to be sampled in evaluation,
        """
        super(PatchMatch, self).__init__()
        self.patchmatch_iteration = patchmatch_iteration
        self.patchmatch_interval_scale = patchmatch_interval_scale
        self.propa_num_feature = num_feature
        # group wise correlation
        self.G = G
        self.stage = stage
        self.dilation = propagation_out_range
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # Using dictionary instead of Enum since TorchScript cannot recognize and export it correctly
        self.grid_type = {"propagation": 1, "evaluation": 2}

        self.depth_initialization = DepthInitialization(patchmatch_num_sample)
        self.propagation = Propagation()
        self.evaluation = Evaluation(self.G)
        # adaptive propagation: last iteration on stage 1 does not have propagation,
        # but we still define this for TorchScript export compatibility
        self.propa_conv = nn.Conv2d(
            in_channels=self.propa_num_feature,
            out_channels=max(2 * self.propagate_neighbors, 1),
            kernel_size=3,
            stride=1,
            padding=self.dilation,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.propa_conv.weight, 0.0)
        nn.init.constant_(self.propa_conv.bias, 0.0)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(
            in_channels=self.propa_num_feature,
            out_channels=2 * self.evaluate_neighbors,
            kernel_size=3,
            stride=1,
            padding=self.dilation,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.eval_conv.weight, 0.0)
        nn.init.constant_(self.eval_conv.bias, 0.0)
        self.feature_weight_net = FeatureWeightNet(self.evaluate_neighbors, self.G)

    def get_grid(
        self, grid_type: int, batch: int, height: int, width: int, offset: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Compute the offset for adaptive propagation or spatial cost aggregation in adaptive evaluation

        Args:
            grid_type: type of grid - propagation (1) or evaluation (2)
            batch: batch size
            height: grid height
            width: grid width
            offset: grid offset
            device: device on which to place tensor

        Returns:
            generated grid: in the shape of [batch, propagate_neighbors*H, W, 2]
        """

        if grid_type == self.grid_type["propagation"]:
            if self.propagate_neighbors == 4:  # if 4 neighbors to be sampled in propagation
                original_offset = [[-self.dilation, 0], [0, -self.dilation], [0, self.dilation], [self.dilation, 0]]
            elif self.propagate_neighbors == 8:  # if 8 neighbors to be sampled in propagation
                original_offset = [
                    [-self.dilation, -self.dilation],
                    [-self.dilation, 0],
                    [-self.dilation, self.dilation],
                    [0, -self.dilation],
                    [0, self.dilation],
                    [self.dilation, -self.dilation],
                    [self.dilation, 0],
                    [self.dilation, self.dilation],
                ]
            elif self.propagate_neighbors == 16:  # if 16 neighbors to be sampled in propagation
                original_offset = [
                    [-self.dilation, -self.dilation],
                    [-self.dilation, 0],
                    [-self.dilation, self.dilation],
                    [0, -self.dilation],
                    [0, self.dilation],
                    [self.dilation, -self.dilation],
                    [self.dilation, 0],
                    [self.dilation, self.dilation],
                ]
                for i in range(len(original_offset)):
                    offset_x, offset_y = original_offset[i]
                    original_offset.append([2 * offset_x, 2 * offset_y])
            else:
                raise NotImplementedError
        elif grid_type == self.grid_type["evaluation"]:
            dilation = self.dilation - 1  # dilation of evaluation is a little smaller than propagation
            if self.evaluate_neighbors == 9:  # if 9 neighbors to be sampled in evaluation
                original_offset = [
                    [-dilation, -dilation],
                    [-dilation, 0],
                    [-dilation, dilation],
                    [0, -dilation],
                    [0, 0],
                    [0, dilation],
                    [dilation, -dilation],
                    [dilation, 0],
                    [dilation, dilation],
                ]
            elif self.evaluate_neighbors == 17:  # if 17 neighbors to be sampled in evaluation
                original_offset = [
                    [-dilation, -dilation],
                    [-dilation, 0],
                    [-dilation, dilation],
                    [0, -dilation],
                    [0, 0],
                    [0, dilation],
                    [dilation, -dilation],
                    [dilation, 0],
                    [dilation, dilation],
                ]
                for i in range(len(original_offset)):
                    offset_x, offset_y = original_offset[i]
                    if offset_x != 0 or offset_y != 0:
                        original_offset.append([2 * offset_x, 2 * offset_y])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        with torch.no_grad():
            y_grid, x_grid = torch.meshgrid(
                [
                    torch.arange(0, height, dtype=torch.float32, device=device),
                    torch.arange(0, width, dtype=torch.float32, device=device),
                ]
            )
            y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list = []
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            offset_x = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
            offset_y = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)
            xy_list.append((xy + torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))

        xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

        del xy_list
        del x_grid
        del y_grid

        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized
        del y_normalized
        return grid.view(batch, len(original_offset) * height, width, 2)

    def forward(
        self,
        ref_feature: torch.Tensor,
        src_features: List[torch.Tensor],
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
        depth: torch.Tensor,
        view_weights: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward method for PatchMatch

        Args:
            ref_feature: feature from reference view, (B, C, H, W)
            src_features: features from (Nview-1) source views, (Nview-1) * (B, C, H, W), where Nview is the number of
                input images (or views) of PatchmatchNet
            ref_proj: projection matrix of reference view, (B, 4, 4)
            src_projs: source matrices of source views, (Nview-1) * (B, 4, 4), where Nview is the number of input
                images (or views) of PatchmatchNet
            depth_min: minimum virtual depth, (B,)
            depth_max: maximum virtual depth, (B,)
            depth: current depth map, (B,1,H,W) or None
            view_weights: Tensor to store weights of source views, in shape of (B,Nview-1,H,W),
                Nview-1 represents the number of source views

        Returns:
            depth_samples: list of depth maps from each patchmatch iteration, Niter * (B,1,H,W)
            score: evaluted probabilities, (B,Ndepth,H,W)
            view_weights: Tensor to store weights of source views, in shape of (B,Nview-1,H,W),
                Nview-1 represents the number of source views
        """
        device = ref_feature.device
        batch, _, height, width = ref_feature.size()

        # the learned additional 2D offsets for adaptive propagation
        propa_grid = torch.empty(0, device=device)
        if self.propagate_neighbors > 0 and not (self.stage == 1 and self.patchmatch_iteration == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            propa_offset = self.propa_conv(ref_feature).view(batch, 2 * self.propagate_neighbors, height * width)
            propa_grid = self.get_grid(self.grid_type["propagation"], batch, height, width, propa_offset, device)

        # the learned additional 2D offsets for adaptive spatial cost aggregation (adaptive evaluation)
        eval_offset = self.eval_conv(ref_feature).view(batch, 2 * self.evaluate_neighbors, height * width)
        eval_grid = self.get_grid(self.grid_type["evaluation"], batch, height, width, eval_offset, device)

        # [B, evaluate_neighbors, H, W]
        feature_weight = self.feature_weight_net(ref_feature.detach(), eval_grid)
        depth_sample = depth
        del depth

        score = torch.empty(0, device=device)
        depth_samples = []
        for iter in range(1, self.patchmatch_iteration + 1):
            is_inverse = self.stage == 1 and iter == self.patchmatch_iteration

            # first iteration on stage 3, random initialization (depth is empty), no adaptive propagation
            # subsequent iterations, local perturbation based on previous result, [B,Ndepth,H,W]
            depth_sample = self.depth_initialization(
                min_depth=depth_min,
                max_depth=depth_max,
                height=height,
                width=width,
                depth_interval_scale=self.patchmatch_interval_scale,
                device=device,
                depth=depth_sample
            )

            # adaptive propagation
            if self.propagate_neighbors > 0 and not (self.stage == 1 and iter == self.patchmatch_iteration):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                depth_sample = self.propagation(depth_sample=depth_sample, grid=propa_grid)

            # weights for adaptive spatial cost aggregation in adaptive evaluation, [B,Ndepth,N_neighbors_eval,H,W]
            weight = depth_weight(
                depth_sample=depth_sample.detach(),
                depth_min=depth_min,
                depth_max=depth_max,
                grid=eval_grid.detach(),
                patchmatch_interval_scale=self.patchmatch_interval_scale,
                neighbors=self.evaluate_neighbors,
            ) * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)  # [B,Ndepth,1,H,W]

            # evaluation, outputs regressed depth map and pixel-wise view weights which will
            # be used for subsequent iterations
            depth_sample, score, view_weights = self.evaluation(
                ref_feature=ref_feature,
                src_features=src_features,
                ref_proj=ref_proj,
                src_projs=src_projs,
                depth_sample=depth_sample,
                grid=eval_grid,
                weight=weight,
                view_weights=view_weights,
                is_inverse=is_inverse,
            )

            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)

        return depth_samples, score, view_weights


class SimilarityNet(nn.Module):
    """Similarity Net, used in Evaluation module (adaptive evaluation step)
    1. Do 1x1x1 convolution on aggregated cost [B, G, Ndepth, H, W] among all the source views,
        where G is the number of groups
    2. Perform adaptive spatial cost aggregation to get final cost (scores)
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(SimilarityNet, self).__init__()

        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.similarity = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: torch.Tensor, grid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward method for SimilarityNet

        Args:
            x1: [B, G, Ndepth, H, W], where G is the number of groups, aggregated cost among all the source views with
                pixel-wise view weight
            grid: position of sampling points in adaptive spatial cost aggregation, (B, evaluate_neighbors*H, W, 2)
            weight: weight of sampling points in adaptive spatial cost aggregation, combination of
                feature weight and depth weight, [B,Ndepth,1,H,W]

        Returns:
            final cost: in the shape of [B,Ndepth,H,W]
        """

        batch, G, num_depth, height, width = x1.size()
        num_neighbors = grid.size()[1] // height

        # [B,Ndepth,num_neighbors,H,W]
        x1 = F.grid_sample(
            input=self.similarity(self.conv1(self.conv0(x1))).squeeze(1),
            grid=grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        ).view(batch, num_depth, num_neighbors, height, width)

        return torch.sum(x1 * weight, dim=2)


class FeatureWeightNet(nn.Module):
    """FeatureWeight Net: Called at the beginning of patchmatch, to calculate feature weights based on similarity of
    features of sampling points and center pixel. The feature weights is used to implement adaptive spatial
    cost aggregation.
    """

    def __init__(self, neighbors: int = 9, G: int = 8) -> None:
        """Initialize method

        Args:
            neighbors: number of neighbors to be sampled
            G: the feature channels of input will be divided evenly into G groups
        """
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = G

        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.similarity = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.output = nn.Sigmoid()

    def forward(self, ref_feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Forward method for FeatureWeightNet

        Args:
            ref_feature: reference feature map, [B,C,H,W]
            grid: position of sampling points in adaptive spatial cost aggregation, (B, evaluate_neighbors*H, W, 2)

        Returns:
            weight based on similarity of features of sampling points and center pixel, [B,Neighbor,H,W]
        """
        batch, feature_channel, height, width = ref_feature.size()

        weight = F.grid_sample(
            ref_feature, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).view(batch, self.G, feature_channel // self.G, self.neighbors, height, width)

        # [B,G,C//G,H,W]
        ref_feature = ref_feature.view(batch, self.G, feature_channel // self.G, height, width).unsqueeze(3)
        # [B,G,Neighbor,H,W]
        weight = (weight * ref_feature).mean(2)
        # [B,Neighbor,H,W]
        return self.output(self.similarity(self.conv1(self.conv0(weight))).squeeze(1))


def depth_weight(
    depth_sample: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    grid: torch.Tensor,
    patchmatch_interval_scale: float,
    neighbors: int,
) -> torch.Tensor:
    """Calculate depth weight
    1. Adaptive spatial cost aggregation
    2. Weight based on depth difference of sampling points and center pixel

    Args:
        depth_sample: sample depth map, (B,Ndepth,H,W)
        depth_min: minimum virtual depth, (B,)
        depth_max: maximum virtual depth, (B,)
        grid: position of sampling points in adaptive spatial cost aggregation, (B, neighbors*H, W, 2)
        patchmatch_interval_scale: patchmatch interval scale,
        neighbors: number of neighbors to be sampled in evaluation

    Returns:
        depth weight
    """
    batch, num_depth, height, width = depth_sample.size()
    inverse_depth_min = 1.0 / depth_min
    inverse_depth_max = 1.0 / depth_max

    # normalization
    x = 1.0 / depth_sample
    del depth_sample
    x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min - inverse_depth_max).view(batch, 1, 1, 1)

    x1 = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False
    ).view(batch, num_depth, neighbors, height, width)
    del grid

    # [B,Ndepth,N_neighbors,H,W]
    x1 = torch.abs(x1 - x.unsqueeze(2)) / patchmatch_interval_scale
    del x

    # sigmoid output approximate to 1 when x=4
    return torch.sigmoid(4.0 - 2.0 * x1.clamp(min=0, max=4)).detach()


class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.

    1. The Pixelwise Net is used in adaptive evaluation step
    2. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    3. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet

        Args:
            x1: pixel-wise view weight, [B, G, Ndepth, H, W], where G is the number of groups
        """
        # [B,1,H,W]
        return torch.max(self.output(self.conv2(self.conv1(self.conv0(x1))).squeeze(1)), dim=1)[0].unsqueeze(1)
