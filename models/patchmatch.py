from .module import *


class DepthInitialization(nn.Module):
    def __init__(self, patch_match_num_sample: int = 1):
        super(DepthInitialization, self).__init__()
        self.patch_match_num_sample = patch_match_num_sample

    def forward(self, batch_size: int, min_depth: float, max_depth: float, height: int, width: int,
                depth_interval_scale: float, device, depth: Tensor):

        if is_empty(depth):
            # first iteration of PatchMatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            inverse_min_depth = 1.0 / min_depth
            inverse_max_depth = 1.0 / max_depth
            patch_match_num_sample = 48
            # [B,num_depth,H,W]
            depth_sample = torch.rand((batch_size, patch_match_num_sample, height, width), device=device, dtype=torch.float32) + \
                           torch.arange(0, patch_match_num_sample, 1, device=device, dtype=torch.float32).view(1, patch_match_num_sample, 1, 1)

            depth_sample = inverse_max_depth + depth_sample / patch_match_num_sample * (inverse_min_depth - inverse_max_depth)

            depth_sample = 1.0 / depth_sample

            return depth_sample

        elif self.patch_match_num_sample == 1:
            # other PatchMatch, local perturbation is performed based on previous result
            # uniform samples in an inverse depth range
            return depth.detach()
        else:
            inverse_min_depth = 1.0 / min_depth
            inverse_max_depth = 1.0 / max_depth

            depth_sample = torch.arange(-self.patch_match_num_sample // 2, self.patch_match_num_sample // 2, 1,
                                        device=device, dtype=torch.float32).view(1, self.patch_match_num_sample, 1, 1).repeat(batch_size, 1, height, width)
            inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_interval_scale
            depth_sample = 1.0 / depth.detach() + inverse_depth_interval * depth_sample

            depth_clamped = []
            del depth
            for k in range(batch_size):
                depth_clamped.append(
                    torch.clamp(depth_sample[k], min=inverse_max_depth, max=inverse_min_depth).unsqueeze(0))

            depth_sample = 1.0 / torch.cat(depth_clamped, dim=0)
            del depth_clamped

            return depth_sample


class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()

    def forward(self, depth: Tensor, grid: Tensor):
        # [B,D,H,W]
        batch_size, num_depth, height, width = depth.size()
        num_neighbors = grid.size()[1] // height
        prop_depth = nnfun.grid_sample(depth[:, num_depth // 2, :, :].unsqueeze(1), grid, mode='bilinear', padding_mode='border')
        del grid
        prop_depth = prop_depth.view(batch_size, num_neighbors, height, width)

        propagate_depth, _ = torch.sort(torch.cat((depth, prop_depth), dim=1), dim=1)
        del prop_depth

        return propagate_depth


class Evaluation(nn.Module):
    def __init__(self, group_size: int = 8, stage: int = 3, evaluate_neighbors: int = 9, iterations: int = 2):
        super(Evaluation, self).__init__()

        self.iterations = iterations

        self.group_size = group_size
        self.stage = stage
        if self.stage == 3:
            self.pixel_wise_net = PixelwiseNet(self.group_size)

        self.similarity_net = SimilarityNet(self.group_size, evaluate_neighbors)

    def forward(self, ref_feature: Tensor, src_features: Tensor, ref_proj: Tensor, src_projs: Tensor,
                depth_sample: Tensor, iteration: int,
                grid: Tensor, weight: Tensor, view_weights: Tensor):

        num_src_features = len(src_features)
        num_src_projs = len(src_projs)
        batch, feature_channel, height, width = ref_feature.size()
        device = ref_feature.get_device()

        num_depth = depth_sample.size()[1]
        assert num_src_features == num_src_projs, "PatchMatch Evaluation: Different number of images and projection matrices"
        if not is_empty(view_weights):
            assert num_src_features == view_weights.size()[
                1], "PatchMatch Evaluation: Different number of images and view weights"

        pixel_wise_weight_sum = 0

        ref_feature = ref_feature.view(batch, self.group_size, feature_channel // self.group_size, height, width)

        similarity_sum = 0

        if self.stage == 3 and is_empty(view_weights):
            view_weights = []
            for src_feature, src_proj in zip(src_features, src_projs):

                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)
                warped_feature = warped_feature.view(batch, self.group_size, feature_channel // self.group_size,
                                                     num_depth, height, width)
                # group-wise correlation
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # pixel-wise view weight
                view_weight = self.pixel_wise_net(similarity)
                view_weights.append(view_weight)

                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, G, num_depth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B,1,1,H,W]
                else:
                    similarity_sum += similarity * view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)

                del warped_feature
                del src_feature
                del src_proj
                del similarity
                del view_weight
            del src_features
            del src_projs
            view_weights = torch.cat(view_weights, dim=1)  # [B,4,H,W], 4 is the number of source views
            # aggregated matching cost across all the source views
            similarity = similarity_sum.div_(pixel_wise_weight_sum)
            del ref_feature
            del pixel_wise_weight_sum
            del similarity_sum
            # adaptive spatial cost aggregation
            score = self.similarity_net(similarity, grid, weight)
            del similarity
            del grid
            del weight

            # apply softmax to get probability
            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)

            # depth regression: expectation
            depth_sample = torch.sum(depth_sample * score, dim=1)

            return depth_sample, score, view_weights.detach()
        else:
            i = 0
            for src_feature, src_proj in zip(src_features, src_projs):
                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)
                warped_feature = warped_feature.view(batch, self.group_size, feature_channel // self.group_size,
                                                     num_depth, height, width)
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # reuse the pixel-wise view weight from first iteration of PatchMatch on stage 3
                view_weight = view_weights[:, i].unsqueeze(1)  # [B,1,H,W]
                i = i + 1

                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, G, num_depth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B,1,1,H,W]
                else:
                    similarity_sum += similarity * view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)

                del warped_feature
                del src_feature
                del src_proj
                del similarity
                del view_weight
            del src_features
            del src_projs

            # [B, G, num_depth, H, W]
            similarity = similarity_sum.div_(pixel_wise_weight_sum)

            del ref_feature
            del pixel_wise_weight_sum
            del similarity_sum

            score = self.similarity_net(similarity, grid, weight)
            del similarity
            del grid
            del weight

            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)

            if self.stage == 1 and iteration == self.iterations:
                # depth regression: inverse depth regression
                depth_index = torch.arange(0, num_depth, 1, device=device, dtype=torch.float32).view(1, num_depth, 1, 1)
                depth_index = torch.sum(depth_index * score, dim=1)

                inverse_min_depth = 1.0 / depth_sample[:, -1, :, :]
                inverse_max_depth = 1.0 / depth_sample[:, 0, :, :]
                depth_sample = inverse_max_depth + depth_index / (num_depth - 1) * \
                               (inverse_min_depth - inverse_max_depth)
                depth_sample = 1.0 / depth_sample

                return depth_sample, score, view_weights

            # depth regression: expectation
            else:
                depth_sample = torch.sum(depth_sample * score, dim=1)

                return depth_sample, score, view_weights


# compute the offset for adaptive propagation
def get_grid(orig_offsets, batch_size: int, height: int, width: int, offset: Tensor, device):
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


class PatchMatch(nn.Module):
    def __init__(self, propagation_out_range: int, iterations: int, patch_match_num_sample: int,
                 patch_match_interval_scale: float, num_feature: int, group_size: int, propagate_neighbors: int,
                 evaluate_neighbors: int, stage: int):
        super(PatchMatch, self).__init__()
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.iterations = iterations
        self.stage = stage
        self.patch_match_interval_scale = patch_match_interval_scale

        self.propagation_offset = self.get_propagation_grid_offset(propagation_out_range)
        self.evaluation_offset = self.get_evaluation_grid_offset(propagation_out_range - 1)
        self.depth_initialization = DepthInitialization(patch_match_num_sample)
        self.propagation = Propagation()
        self.evaluation = Evaluation(group_size, self.stage, self.evaluate_neighbors, self.iterations)
        # adaptive propagation
        if self.propagate_neighbors > 0 and not (self.stage == 1 and self.iterations == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            self.propa_conv = nn.Conv2d(num_feature, 2 * self.propagate_neighbors, kernel_size=3, stride=1,
                                        padding=propagation_out_range, dilation=propagation_out_range, bias=True)
            nn.init.constant_(self.propa_conv.weight, 0.)
            nn.init.constant_(self.propa_conv.bias, 0.)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(num_feature, 2 * self.evaluate_neighbors, kernel_size=3, stride=1,
                                   padding=propagation_out_range, dilation=propagation_out_range, bias=True)
        nn.init.constant_(self.eval_conv.weight, 0.)
        nn.init.constant_(self.eval_conv.bias, 0.)
        self.feature_weight_net = FeatureWeightNet(self.evaluate_neighbors, group_size)

    def get_propagation_grid_offset(self, dilation: int):
        if self.propagate_neighbors == 0:
            offset = []
        elif self.propagate_neighbors == 4:
            offset = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
        elif self.propagate_neighbors == 8:
            offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, dilation],
                      [dilation, -dilation], [dilation, 0], [dilation, dilation]]
        elif self.propagate_neighbors == 16:
            offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, dilation],
                      [dilation, -dilation], [dilation, 0], [dilation, dilation]]
            for i in range(len(offset)):
                offset_x, offset_y = offset[i]
                offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError

        return offset

    def get_evaluation_grid_offset(self, dilation: int):
        if self.evaluate_neighbors == 9:
            offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, 0],
                      [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
        elif self.evaluate_neighbors == 17:
            offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation], [0, -dilation], [0, 0],
                      [0, dilation], [dilation, -dilation], [dilation, 0], [dilation, dilation]]
            for i in range(len(offset)):
                offset_x, offset_y = offset[i]
                if offset_x != 0 or offset_y != 0:
                    offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError

        return offset

    def forward(self, ref_feature: Tensor, src_features: Tensor, ref_proj: Tensor, src_projs: Tensor, depth_min: float,
                depth_max: float, depth: Tensor, view_weights: Tensor):
        depth_sample = depth
        score = torch.Tensor()
        del depth

        device = ref_feature.get_device()
        batch_size, _, height, width = ref_feature.size()

        # the learned additional 2D offsets for adaptive propagation
        propagation_grid = torch.Tensor()
        if self.propagate_neighbors > 0 and not (self.stage == 1 and self.iterations == 1):
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            prop_offset = self.propa_conv(ref_feature)
            prop_offset = prop_offset.view(batch_size, 2 * self.propagate_neighbors, height * width)
            propagation_grid = get_grid(self.propagation_offset, batch_size, height, width, prop_offset, device)

        # the learned additional 2D offsets for adaptive spatial cost aggregation (adaptive evaluation)
        eval_offset = self.eval_conv(ref_feature)
        eval_offset = eval_offset.view(batch_size, 2 * self.evaluate_neighbors, height * width)
        eval_grid = get_grid(self.evaluation_offset, batch_size, height, width, eval_offset, device)

        feature_weight = self.feature_weight_net(ref_feature.detach(), eval_grid)

        # patch-match iterations with local perturbation based on previous result
        for iteration in range(1, self.iterations + 1):
            # local perturbation based on previous result
            depth_sample = self.depth_initialization(batch_size, depth_min, depth_max, height, width,
                                                     self.patch_match_interval_scale, device, depth_sample)

            # adaptive propagation
            if self.propagate_neighbors > 0 and not (self.stage == 1 and iteration == self.iterations):
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                depth_sample = self.propagation(depth_sample, propagation_grid)

            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = depth_weight(depth_sample.detach(), depth_min, depth_max, eval_grid.detach(),
                                  self.patch_match_interval_scale, self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)

            # evaluation, outputs regressed depth map and pixel-wise view weights used for subsequent iterations
            depth_sample, score, view_weights = self.evaluation(ref_feature, src_features, ref_proj, src_projs,
                                                                depth_sample, iteration, eval_grid, weight, view_weights)

            depth_sample = depth_sample.unsqueeze(1)

        return depth_sample, score, view_weights


# first, do convolution on aggregated cost among all the source views
# second, perform adaptive spatial cost aggregation to get final cost
class SimilarityNet(nn.Module):
    def __init__(self, group_size: int, neighbors: int = 9):
        super(SimilarityNet, self).__init__()
        self.neighbors = neighbors

        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: Tensor, grid: Tensor, weight: Tensor):
        # x1: [B, G, num_depth, H, W], aggregated cost among all the source views with pixel-wise view weight
        # grid: position of sampling points in adaptive spatial cost aggregation
        # weight: weight of sampling points in adaptive spatial cost aggregation, combination of 
        # feature weight and depth weight

        batch, group_size, num_depth, height, width = x1.size()

        x1 = self.similarity(self.conv1(self.conv0(x1))).squeeze(1)

        x1 = nnfun.grid_sample(x1, grid, mode='bilinear', padding_mode='border')

        # [B,num_depth,9,H,W]
        x1 = x1.view(batch, num_depth, self.neighbors, height, width)

        return torch.sum(x1 * weight, dim=2)


# adaptive spatial cost aggregation
# weight based on similarity of features of sampling points and center pixel
class FeatureWeightNet(nn.Module):
    def __init__(self, neighbors: int = 9, group_size: int = 8):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = group_size

        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.output = nn.Sigmoid()

    def forward(self, ref_feature: Tensor, grid: Tensor):
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation
        batch, feature_channel, height, width = ref_feature.size()

        x = nnfun.grid_sample(ref_feature, grid, mode='bilinear', padding_mode='border')

        # [B,G,C//G,H,W]
        ref_feature = ref_feature.view(batch, self.G, feature_channel // self.G, height, width)

        x = x.view(batch, self.G, feature_channel // self.G, self.neighbors, height, width)
        # [B,G,Neighbor,H,W]
        x = (x * ref_feature.unsqueeze(3)).mean(2)
        del ref_feature
        # [B,Neighbor,H,W]
        x = self.similarity(self.conv1(self.conv0(x))).squeeze(1)

        return self.output(x)


# adaptive spatial cost aggregation
# weight based on depth difference of sampling points and center pixel
def depth_weight(depth_sample: Tensor, depth_min: float, depth_max: float, grid: Tensor,
                 patch_match_interval_scale: float, neighbors: int):
    # grid: position of sampling points in adaptive spatial cost aggregation
    batch, num_depth, height, width = depth_sample.size()
    # normalization
    x = 1.0 / depth_sample
    del depth_sample
    inverse_depth_min = 1.0 / depth_min
    inverse_depth_max = 1.0 / depth_max
    x = (x - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)

    x1 = nnfun.grid_sample(x.float(), grid, mode='bilinear', padding_mode='border')
    del grid
    x1 = x1.view(batch, num_depth, neighbors, height, width)

    # [B,num_depth,N_neighbors,H,W]
    x1 = torch.abs(x1 - x.unsqueeze(2)) / patch_match_interval_scale
    del x
    x1 = torch.clamp(x1, min=0, max=4)
    # sigmoid output approximate to 1 when x=4
    x1 = (-x1 + 2) * 2
    output = nn.Sigmoid()
    x1 = output(x1)

    return x1.detach()


# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, group_size: int):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(group_size, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: Tensor):
        # x1: [B, G, num_depth, H, W]

        # [B, num_depth, H, W]
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1)

        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]

        return output.unsqueeze(1)
