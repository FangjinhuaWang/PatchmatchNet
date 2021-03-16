from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional
from torch import Tensor

from models.module import ConvBnReLU3D, differentiable_warping, is_empty


class DepthInitialization(nn.Module):
    def __init__(self, num_samples: int, depth_interval_scale: float):
        super(DepthInitialization, self).__init__()
        self.num_samples = num_samples
        self.depth_interval_scale = depth_interval_scale

    def forward(self, batch_size: int, min_depth: float, max_depth: float, height: int, width: int,
                device: torch.device, depth: Tensor) -> Tensor:

        inverse_min_depth = 1.0 / min_depth
        inverse_max_depth = 1.0 / max_depth
        if is_empty(depth):
            # first iteration of PatchMatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            num_samples = 48
            # [B,num_depth,H,W]
            depth_sample = torch.rand((batch_size, num_samples, height, width), device=device, dtype=torch.float32) + \
                torch.arange(0, num_samples, 1, device=device, dtype=torch.float32).view(1, num_samples, 1, 1)

            return 1.0 / (inverse_max_depth + depth_sample / num_samples * (
                        inverse_min_depth - inverse_max_depth))

        elif self.num_samples == 1:
            # other PatchMatch, local perturbation is performed based on previous result
            # uniform samples in an inverse depth range
            return depth.detach()
        else:
            depth_sample = torch.arange(
                -self.num_samples // 2, self.num_samples // 2, 1, device=device, dtype=torch.float32)
            depth_sample = depth_sample.view(1, self.num_samples, 1, 1).repeat(batch_size, 1, height, width)
            depth_sample = 1.0 / depth.detach() + (
                    inverse_min_depth - inverse_max_depth) * self.depth_interval_scale * depth_sample

            return 1.0 / depth_sample.clamp(min=inverse_max_depth, max=inverse_min_depth)


class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()

    def forward(self, depth: Tensor, grid: Tensor) -> Tensor:
        # [B,D,H,W]
        batch_size, num_depth, height, width = depth.size()
        num_neighbors = grid.size()[1] // height
        prop_depth = nn.functional.grid_sample(depth[:, num_depth // 2, :, :].unsqueeze(1), grid, mode='bilinear',
                                               padding_mode='border', align_corners=False)
        prop_depth = prop_depth.view(batch_size, num_neighbors, height, width)
        return torch.sort(torch.cat((depth, prop_depth), dim=1), dim=1)[0]


class Evaluation(nn.Module):
    def __init__(self, group_size: int):
        super(Evaluation, self).__init__()

        self.group_size = group_size
        self.pixel_wise_net = PixelwiseNet(group_size)
        self.similarity_net = SimilarityNet(group_size)

    def forward(self, ref_feature: Tensor, src_features: List[Tensor], ref_proj: Tensor, src_projs: List[Tensor],
                depth_sample: Tensor, grid: Tensor, weight: Tensor, view_weights: Tensor, is_inverse: bool) \
            -> Tuple[Tensor, Tensor, Tensor]:

        device = ref_feature.device
        batch_size, num_channels, height, width = ref_feature.size()
        num_depth = depth_sample.size()[1]
        no_input_weights = is_empty(view_weights)
        view_weights_new = []

        assert len(src_features) == len(src_projs), 'PatchMatch Eval: Num images != num projection matrices'
        if not is_empty(view_weights):
            assert len(src_features) == view_weights.size()[1], 'PatchMatch Eval: Num images != num weights'

        ref_feature = ref_feature.view(batch_size, self.group_size, num_channels // self.group_size, 1, height, width)
        weight_sum = torch.zeros((batch_size, 1, 1, height, width), dtype=torch.float32, device=device)
        similarity_sum = torch.zeros((batch_size, self.group_size, num_depth, height, width), dtype=torch.float32,
                                     device=device)
        i = 0
        # print('Evaluation sizes')
        for src_feature, src_proj in zip(src_features, src_projs):
            warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)
            # print(src_feature.size())
            # print(depth_sample.size())
            # print(warped_feature.size())
            warped_feature = warped_feature.view(batch_size, self.group_size, num_channels // self.group_size,
                                                 num_depth, height, width)
            similarity = (warped_feature * ref_feature).mean(2)
            if no_input_weights:
                # pixel-wise view weight
                view_weight = self.pixel_wise_net(similarity)
            else:
                # reuse the pixel-wise view weight from first iteration of PatchMatch on stage 3
                view_weight = view_weights[:, i].unsqueeze(1)  # [B,1,H,W]
                i = i + 1

            view_weights_new.append(view_weight)
            similarity_sum += similarity * view_weight.unsqueeze(1)  # [B, G, num_depth, H, W]
            weight_sum += view_weight.unsqueeze(1)  # [B,1,1,H,W]

        del src_features
        del src_projs
        del ref_feature

        # aggregated matching cost across all the source views, adaptive spatial cost aggregation,
        # and log_softmax to get probability
        score = torch.exp(torch.log_softmax(self.similarity_net(similarity_sum.div_(weight_sum), grid, weight), dim=1))
        del weight_sum
        del similarity_sum
        del grid
        del weight

        if is_inverse:
            # depth regression: inverse depth regression
            depth_index = torch.arange(0, num_depth, 1, device=device, dtype=torch.float32).view(1, num_depth, 1, 1)
            depth_index = torch.sum(depth_index * score, dim=1)
            inv_min_depth = 1.0 / depth_sample[:, -1, :, :]
            inv_max_depth = 1.0 / depth_sample[:, 0, :, :]
            depth_sample = 1.0 / (inv_max_depth + depth_index / (num_depth - 1) * (inv_min_depth - inv_max_depth))
        else:
            # depth regression: expectation
            depth_sample = torch.sum(depth_sample * score, dim=1)

        return depth_sample, score, torch.cat(view_weights_new, dim=1).detach()


class PatchMatch(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, propagation_out_range: int, iterations: int, patch_match_num_sample: int,
                 patch_match_interval_scale: float, num_feature: int, group_size: int, propagate_neighbors: int,
                 evaluate_neighbors: int, stage: int):
        super(PatchMatch, self).__init__()
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.iterations = iterations
        self.stage = stage
        self.patch_match_interval_scale = patch_match_interval_scale
        self.dilation = propagation_out_range
        self.depth_initialization = DepthInitialization(patch_match_num_sample, patch_match_interval_scale)
        self.propagation = Propagation()
        self.evaluation = Evaluation(group_size)
        # adaptive propagation
        if self.propagate_neighbors > 0 and not (self.stage == 1 and self.iterations == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            self.propa_conv = nn.Conv2d(num_feature, 2 * self.propagate_neighbors, kernel_size=3, stride=1,
                                        padding=propagation_out_range, dilation=propagation_out_range, bias=True)
            nn.init.constant_(self.propa_conv.weight, 0.)
            nn.init.constant_(self.propa_conv.bias, 0.)
        else:
            self.propa_conv = nn.Conv2d(num_feature, 1, kernel_size=3)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(num_feature, 2 * self.evaluate_neighbors, kernel_size=3, stride=1,
                                   padding=propagation_out_range, dilation=propagation_out_range, bias=True)
        nn.init.constant_(self.eval_conv.weight, 0.)
        nn.init.constant_(self.eval_conv.bias, 0.)
        self.feature_weight_net = FeatureWeightNet(self.evaluate_neighbors, group_size)

    # compute the offset for adaptive propagation
    def get_grid(self, is_eval: bool, batch_size: int, height: int, width: int, offset: Tensor, device: torch.device) \
            -> Tensor:
        # orig_offsets: List[List[int]] = []
        if is_eval:
            dilation = self.dilation - 1
            if self.evaluate_neighbors == 9:
                orig_offsets = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, 0],
                                [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
            elif self.evaluate_neighbors == 17:
                orig_offsets = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, 0],
                                [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
                for i in range(len(orig_offsets)):
                    offset_x, offset_y = orig_offsets[i]
                    if offset_x != 0 or offset_y != 0:
                        orig_offsets.append([2 * offset_x, 2 * offset_y])
            else:
                raise NotImplementedError
        else:
            dilation = self.dilation
            if self.propagate_neighbors == 0:
                orig_offsets: List[List[int]] = []
            elif self.propagate_neighbors == 4:
                orig_offsets = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
            elif self.propagate_neighbors == 8:
                orig_offsets = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation],
                                [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
            elif self.propagate_neighbors == 16:
                orig_offsets = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation],
                                [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
                for i in range(len(orig_offsets)):
                    offset_x, offset_y = orig_offsets[i]
                    orig_offsets.append([2 * offset_x, 2 * offset_y])
            else:
                raise NotImplementedError

        with torch.no_grad():
            y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                                             torch.arange(0, width, dtype=torch.float32, device=device)])
            y_grid, x_grid = y_grid.contiguous(), x_grid.contiguous()
            y_grid, x_grid = y_grid.view(height * width), x_grid.view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch_size, 1, 1)  # [B, 2, H*W]

        xy_list = []
        for i in range(len(orig_offsets)):
            original_offset_y, original_offset_x = orig_offsets[i]

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
        grid = grid.view(batch_size, len(orig_offsets) * height, width, 2)
        return grid

    def forward(self, ref_feature: Tensor, src_features: List[Tensor], ref_proj: Tensor, src_projs: List[Tensor],
                depth_min: float, depth_max: float, depth: Tensor, weights: Tensor, all_samples: bool = False) \
            -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        score = torch.empty(0)
        samples: List[Tensor] = []

        device = ref_feature.device
        batch_size, _, height, width = ref_feature.size()

        # the learned additional 2D offsets for adaptive propagation
        propagation_grid = torch.empty(0)
        if self.propagate_neighbors > 0 and not (self.stage == 1 and self.iterations == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            prop_offset = self.propa_conv(ref_feature)
            prop_offset = prop_offset.view(batch_size, 2 * self.propagate_neighbors, height * width)
            propagation_grid = self.get_grid(False, batch_size, height, width, prop_offset, device)

        # the learned additional 2D offsets for adaptive spatial cost aggregation (adaptive evaluation)
        eval_offset = self.eval_conv(ref_feature)
        eval_offset = eval_offset.view(batch_size, 2 * self.evaluate_neighbors, height * width)
        eval_grid = self.get_grid(True, batch_size, height, width, eval_offset, device)

        feature_weight = self.feature_weight_net(ref_feature.detach(), eval_grid)

        # patch-match iterations with local perturbation based on previous result
        for iteration in range(1, self.iterations + 1):
            # local perturbation based on previous result
            depth = self.depth_initialization(batch_size, depth_min, depth_max, height, width, device, depth)

            # adaptive propagation
            if self.propagate_neighbors > 0 and not (self.stage == 1 and iteration == self.iterations):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                depth = self.propagation(depth, propagation_grid)

            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = depth_weight(depth.detach(), depth_min, depth_max, eval_grid.detach(),
                                  self.patch_match_interval_scale, self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2, keepdim=True)

            # evaluation, outputs regressed depth map and pixel-wise view weights used for subsequent iterations
            is_inverse = self.stage == 1 and iteration == self.iterations
            depth, score, weights = self.evaluation(ref_feature, src_features, ref_proj, src_projs, depth,
                                                    eval_grid, weight, weights, is_inverse)

            depth = depth.unsqueeze(1)
            if all_samples:
                samples.append(depth)

        return depth, score, weights, samples


# first, do convolution on aggregated cost among all the source views
# second, perform adaptive spatial cost aggregation to get final cost
class SimilarityNet(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, group_size: int):
        super(SimilarityNet, self).__init__()

        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: Tensor, grid: Tensor, weight: Tensor) -> Tensor:
        # x1: [B, G, num_depth, H, W], aggregated cost among all the source views with pixel-wise view weight
        # grid: position of sampling points in adaptive spatial cost aggregation
        # weight: weight of sampling points in adaptive spatial cost aggregation, combination of 
        # feature weight and depth weight

        batch, group_size, num_depth, height, width = x1.size()
        num_neighbors = grid.size()[1] // height
        x1 = nn.functional.grid_sample(self.similarity(self.conv1(self.conv0(x1))).squeeze(1), grid, mode='bilinear',
                                       padding_mode='border', align_corners=False)

        # [B,num_depth,9,H,W]
        x1 = x1.view(batch, num_depth, num_neighbors, height, width)

        return torch.sum(x1 * weight, dim=2)


# adaptive spatial cost aggregation
# weight based on similarity of features of sampling points and center pixel
class FeatureWeightNet(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, neighbors: int, group_size: int):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.group_size = group_size

        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.output = nn.Sigmoid()

    def forward(self, ref_feature: Tensor, grid: Tensor) -> Tensor:
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation
        batch_size, num_channels, height, width = ref_feature.size()

        weight = nn.functional.grid_sample(ref_feature, grid, mode='bilinear', padding_mode='border',
                                           align_corners=False)
        weight = weight.view(batch_size, self.group_size, num_channels // self.group_size, self.neighbors, height,
                             width)

        # [B,G,C//G,H,W]
        ref_feature = ref_feature.view(batch_size, self.group_size, num_channels // self.group_size, height,
                                       width).unsqueeze(3)

        # [B,G,Neighbor,H,W]
        weight = (weight * ref_feature).mean(2)
        del ref_feature
        # [B,Neighbor,H,W]
        return self.output(self.similarity(self.conv1(self.conv0(weight)))).squeeze(1)


# adaptive spatial cost aggregation
# weight based on depth difference of sampling points and center pixel
def depth_weight(depth_sample: Tensor, depth_min: float, depth_max: float, grid: Tensor,
                 patch_match_interval_scale: float, neighbors: int) -> Tensor:
    # grid: position of sampling points in adaptive spatial cost aggregation
    batch, num_depth, height, width = depth_sample.size()
    inverse_depth_min = 1.0 / depth_min
    inverse_depth_max = 1.0 / depth_max

    # normalization
    x = 1.0 / depth_sample
    del depth_sample
    x = (x - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)

    x1 = nn.functional.grid_sample(x.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
    del grid
    x1 = x1.view(batch, num_depth, neighbors, height, width)

    # [B,num_depth,N_neighbors,H,W]
    x1 = torch.abs(x1 - x.unsqueeze(2)) / patch_match_interval_scale
    del x

    return torch.sigmoid((2 - x1.clamp(min=0, max=4)) * 2).detach()


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, group_size: int):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: Tensor) -> Tensor:
        # [B, 1, H,W]
        return torch.max(self.output(self.conv2(self.conv1(self.conv0(x1)))).squeeze(1), dim=1)[0].unsqueeze(1)
