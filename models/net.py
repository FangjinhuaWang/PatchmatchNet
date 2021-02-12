from .patchmatch import *


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

    def forward(self, x):
        output_feature = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature['stage_3'] = self.output1(conv10)

        intra_feat = nnfun.interpolate(conv10, scale_factor=2, mode="bilinear") + self.inner1(conv7)
        del conv7, conv10
        output_feature['stage_2'] = self.output2(intra_feat)

        intra_feat = nnfun.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner2(conv4)
        del conv4
        output_feature['stage_1'] = self.output3(intra_feat)

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

    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1))

        conv0 = self.conv0(img)
        deconv = nnfun.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = nnfun.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)

        return depth


class PatchMatchNet(nn.Module):
    def __init__(self, patch_match_interval_scale, propagation_range, patch_match_iteration, patch_match_num_sample,
                 propagate_neighbors, evaluate_neighbors):
        super(PatchMatchNet, self).__init__()

        self.stages = 4
        self.feature = FeatureNet()
        self.patch_match_num_sample = patch_match_num_sample

        num_features = [8, 16, 32, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        for stage in range(self.stages - 1):

            if stage == 2:
                patch_match = PatchMatch(True, propagation_range[stage], patch_match_iteration[stage],
                                         patch_match_num_sample[stage], patch_match_interval_scale[stage],
                                         num_features[stage + 1], self.G[stage], self.propagate_neighbors[stage],
                                         stage + 1, evaluate_neighbors[stage])
            else:
                patch_match = PatchMatch(False, propagation_range[stage], patch_match_iteration[stage],
                                         patch_match_num_sample[stage], patch_match_interval_scale[stage],
                                         num_features[stage + 1], self.G[stage], self.propagate_neighbors[stage],
                                         stage + 1, evaluate_neighbors[stage])
            setattr(self, f'patchmatch_{stage + 1}', patch_match)

        self.upsample_net = Refinement()

    def forward(self, images, proj_matrices, depth_min, depth_max):

        images_0 = torch.unbind(images['stage_0'], 1)
        images_0_ref = images_0[0]

        self.proj_matrices_0 = torch.unbind(proj_matrices['stage_0'].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices['stage_1'].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices['stage_2'].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices['stage_3'].float(), 1)
        del proj_matrices

        assert len(images_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"

        # step 1. Multi-scale feature extraction
        features = []
        for img in images_0:
            output_feature = self.feature(img)
            features.append(output_feature)
        del images_0
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patch-match
        depth = None
        score = None
        view_weights = None
        depth_patch_match = {}
        refined_depth = {}

        for stage in reversed(range(1, self.stages)):
            src_features_l = [src_fea[f'stage_{stage}'] for src_fea in src_features]
            projs_l = getattr(self, f'proj_matrices_{stage}')
            ref_proj, src_projs = projs_l[0], projs_l[1:]

            depth, score, view_weights = getattr(self, f'patchmatch_{stage}')(ref_feature[f'stage_{stage}'],
                                                                              src_features_l,
                                                                              ref_proj, src_projs,
                                                                              depth_min, depth_max, depth=depth,
                                                                              view_weights=view_weights)

            del src_features_l, ref_proj, src_projs, projs_l

            depth_patch_match[f'stage_{stage}'] = depth

            depth = depth[-1].detach()
            if stage > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = nnfun.interpolate(depth,
                                          scale_factor=2, mode='nearest')
                view_weights = nnfun.interpolate(view_weights,
                                                 scale_factor=2, mode='nearest')

        # step 3. Refinement  
        depth = self.upsample_net(images_0_ref, depth, depth_min, depth_max)
        refined_depth['stage_0'] = depth

        del depth, ref_feature, src_features

        if self.training:
            return {"refined_depth": refined_depth,
                    "depth_patch_match": depth_patch_match,
                    }

        else:
            num_depth = self.patch_match_num_sample[0]
            score_sum4 = 4 * nnfun.avg_pool3d(nnfun.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                              stride=1,
                                              padding=0).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(score, depth_values=torch.arange(num_depth, device=score.device,
                                                                            dtype=torch.float)).long()
            depth_index = torch.clamp(depth_index, 0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = nnfun.interpolate(photometric_confidence,
                                                       scale_factor=2, mode='nearest')
            photometric_confidence = photometric_confidence.squeeze(1)

            return {"refined_depth": refined_depth,
                    "depth_patch_match": depth_patch_match,
                    "photometric_confidence": photometric_confidence,
                    }


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
