from .patchmatch import *
from torch import Tensor
from typing import List


class FeatureNet(nn.Module):
    def __init__(self):
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

    def forward(self, x: Tensor):
        output_feature = [torch.empty(1), torch.empty(1), torch.empty(1), torch.empty(1)]

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature[3] = self.output1(conv10)

        intra_feat = nnfun.interpolate(conv10, scale_factor=2.0, mode='bilinear', align_corners=False) + self.inner1(conv7)
        del conv7
        del conv10
        output_feature[2] = self.output2(intra_feat)

        intra_feat = nnfun.interpolate(intra_feat, scale_factor=2.0, mode='bilinear', align_corners=False) + self.inner2(conv4)
        del conv4
        output_feature[1] = self.output3(intra_feat)

        del intra_feat

        return output_feature


class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8)
        self.res = nn.Conv2d(8, 1, 3, padding=1, bias=False)

    def forward(self, img: Tensor, depth_0: Tensor, depth_min: float, depth_max: float):
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min) / (depth_max - depth_min)

        conv0 = self.conv0(img)
        deconv = nnfun.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv
        del conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = nnfun.interpolate(depth, scale_factor=2.0, mode='bilinear', align_corners=False) + res
        # convert the normalized depth back
        depth = depth * (depth_max - depth_min) + depth_min

        return depth


class PatchMatchNet(nn.Module):
    def __init__(self, patch_match_interval_scale, propagation_range, patch_match_iteration, patch_match_num_sample,
                 propagate_neighbors, evaluate_neighbors):
        super(PatchMatchNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.num_depth = patch_match_num_sample[0]

        num_features = [16, 32, 64]
        # number of groups for group-wise correlation
        num_groups = [4, 8, 8]

        self.patchmatch_1 = PatchMatch(propagation_range[0], patch_match_iteration[0], patch_match_num_sample[0],
                                       patch_match_interval_scale[0], num_features[0], num_groups[0],
                                       propagate_neighbors[0], evaluate_neighbors[0], 1)
        self.patchmatch_2 = PatchMatch(propagation_range[1], patch_match_iteration[1], patch_match_num_sample[1],
                                       patch_match_interval_scale[1], num_features[1], num_groups[1],
                                       propagate_neighbors[1], evaluate_neighbors[1], 2)
        self.patchmatch_3 = PatchMatch(propagation_range[2], patch_match_iteration[2], patch_match_num_sample[2],
                                       patch_match_interval_scale[2], num_features[2], num_groups[2],
                                       propagate_neighbors[2], evaluate_neighbors[2], 3)

        self.upsample_net = Refinement()

    def forward(self, images: Tensor, intrinsics: Tensor, extrinsics: Tensor, depth_params: Tensor):
        depth_min = depth_params[0, 0].item()
        depth_max = depth_params[0, 1].item()
        image_ref = images[:, 0]

        assert images.size()[1] == intrinsics.size()[1], 'Different number of images and intrinsic matrices'
        assert images.size()[1] == extrinsics.size()[1], 'Different number of images and extrinsic matrices'

        # step 1. Multi-scale feature extraction
        features: List[List[Tensor]] = []
        for i in range(images.size()[1]):
            output_feature = self.feature(images[:, i])
            features.append(output_feature)
        ref_feature, src_features = features[0], features[1:]

        # step 2. Learning-based patch-match
        depth = torch.empty(0)
        score = torch.empty(0)
        view_weights = torch.empty(0)

        scale = 0.125
        for stage in range(self.stages - 1, 0, -1):
            src_features_stage = [src_fea[stage] for src_fea in src_features]
            intrinsics_stage = intrinsics.clone()
            intrinsics_stage[:, :, :2] *= scale
            proj = extrinsics.clone()
            proj[:, :, :3, :4] = torch.matmul(intrinsics_stage, extrinsics[:, :, :3, :4])
            proj_stage = torch.unbind(proj, 1)
            ref_proj, src_proj = proj_stage[0], proj_stage[1:]
            scale *= 2.0

            if stage == 3:
                depth, score, view_weights = self.patchmatch_3(ref_feature[stage], src_features_stage, ref_proj, src_proj,
                                                               depth_min, depth_max, depth, view_weights)
            elif stage == 2:
                depth, score, view_weights = self.patchmatch_2(ref_feature[stage], src_features_stage, ref_proj,
                                                               src_proj,
                                                               depth_min, depth_max, depth, view_weights)
            elif stage == 1:
                depth, score, view_weights = self.patchmatch_1(ref_feature[stage], src_features_stage, ref_proj, src_proj,
                                                               depth_min, depth_max, depth, view_weights)

            depth = depth.detach()
            if stage > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = nnfun.interpolate(depth, scale_factor=2.0, mode='bilinear', align_corners=False)
                view_weights = nnfun.interpolate(view_weights, scale_factor=2.0, mode='bilinear', align_corners=False)

        # step 3. Refinement  
        depth = self.upsample_net(image_ref, depth, depth_min, depth_max)

        del ref_feature
        del src_features

        # [B, 1, H, W]
        depth_values = torch.arange(self.num_depth, device=score.device, dtype=torch.float32).view(1, self.num_depth, 1, 1)
        depth_index = torch.sum(score * depth_values, 1, keepdim=True).long().clamp(0, self.num_depth - 1)

        score_sum4 = 4 * nnfun.avg_pool3d(nnfun.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
                                          padding=0).squeeze(1)
        confidence = torch.gather(score_sum4, 1, depth_index)
        confidence = nnfun.interpolate(confidence, scale_factor=2.0, mode='bilinear', align_corners=False)
        confidence = confidence.squeeze(1)

        return depth, confidence


def patch_match_net_loss(depth_patch_match, refined_depth, depth_gt, mask):
    stage = 4
    loss = 0
    for stage in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{stage}']
        mask_l = mask[f'stage_{stage}'] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patch_match_l = depth_patch_match[f'stage_{stage}']
        for i in range(len(depth_patch_match_l)):
            depth1 = depth_patch_match_l[i][mask_l]
            loss = loss + nnfun.smooth_l1_loss(depth1, depth2, reduction='mean')

    stage = 0
    depth_refined_l = refined_depth[f'stage_{stage}']
    depth_gt_l = depth_gt[f'stage_{stage}']
    mask_l = mask[f'stage_{stage}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + nnfun.smooth_l1_loss(depth1, depth2, reduction='mean')

    return loss


class PatchMatchContainer(torch.nn.Module):
    def __init__(self, state):
        super(PatchMatchContainer, self).__init__()
        for key in state:
            setattr(self, key, state[key])
