import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import cv2
import numpy as np


class DepthInitialization(nn.Module):
    def __init__(self, patchmatch_num_sample = 1):
        super(DepthInitialization, self).__init__()
        self.patchmatch_num_sample = patchmatch_num_sample

    
    def forward(self, random_initialization, min_depth, max_depth, height, width, depth_interval_scale, device, 
                depth=None):
        
        
        batch_size = min_depth.size()[0]
        if random_initialization:
            # first iteration of Patchmatch on stage 3, sample in the inverse depth range
            # divide the range into several intervals and sample in each of them
            inverse_min_depth = 1.0 / min_depth
            inverse_max_depth = 1.0 / max_depth
            patchmatch_num_sample = 48 
            # [B,Ndepth,H,W]
            depth_sample = torch.rand((batch_size, patchmatch_num_sample, height, width), device=device) + \
                                torch.arange(0, patchmatch_num_sample, 1, device=device).view(1, patchmatch_num_sample, 1, 1)
       
            depth_sample = inverse_max_depth.view(batch_size,1,1,1) + depth_sample / patchmatch_num_sample * \
                                    (inverse_min_depth.view(batch_size,1,1,1) - inverse_max_depth.view(batch_size,1,1,1))
            
            depth_sample = 1.0 / depth_sample
           
            return depth_sample
            
        else:
            # other Patchmatch, local perturbation is performed based on previous result
            # uniform samples in an inversed depth range
            if self.patchmatch_num_sample == 1:
                return depth.detach()
            else:
                inverse_min_depth = 1.0 / min_depth
                inverse_max_depth = 1.0 / max_depth
                
                depth_sample = torch.arange(-self.patchmatch_num_sample//2, self.patchmatch_num_sample//2, 1, 
                                    device=device).view(1, self.patchmatch_num_sample, 1, 1).repeat(batch_size,
                                    1, height, width).float()
                inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_interval_scale
                inverse_depth_interval = inverse_depth_interval.view(batch_size,1,1,1)
                
                depth_sample = 1.0 / depth.detach() + inverse_depth_interval * depth_sample
                
                depth_clamped = []
                del depth
                for k in range(batch_size):
                    depth_clamped.append(torch.clamp(depth_sample[k], min=inverse_max_depth[k], max=inverse_min_depth[k]).unsqueeze(0))
                
                depth_sample = 1.0 / torch.cat(depth_clamped,dim=0)
                del depth_clamped
                
                return depth_sample
                

class Propagation(nn.Module):
    def __init__(self, neighbors = 16):
        super(Propagation, self).__init__()
        self.neighbors = neighbors
        
        
    
    def forward(self, batch, height, width, depth_sample, grid, depth_min, depth_max, depth_interval_scale):
        # [B,D,H,W]
        num_depth = depth_sample.size()[1]    
        propogate_depth = depth_sample.new_empty(batch, num_depth + self.neighbors, height, width)
        propogate_depth[:,0:num_depth,:,:] = depth_sample
        
        
        propogate_depth_sample = F.grid_sample(depth_sample[:, num_depth // 2,:,:].unsqueeze(1), 
                                    grid, 
                                    mode='bilinear',
                                    padding_mode='border')
        del grid
        propogate_depth_sample = propogate_depth_sample.view(batch, self.neighbors, height, width)
        
        propogate_depth[:,num_depth:,:,:] = propogate_depth_sample
        del propogate_depth_sample
        
        # sort
        propogate_depth, _ = torch.sort(propogate_depth, dim=1)
        
        return propogate_depth
        
        
        

class Evaluation(nn.Module):
    def __init__(self,  G=8, stage=3, evaluate_neighbors=9, iterations=2):
        super(Evaluation, self).__init__()
        
        self.iterations = iterations
        
        self.G = G
        self.stage = stage
        if self.stage == 3:
            self.pixel_wise_net = PixelwiseNet(self.G)
        
        self.similarity_net = SimilarityNet(self.G, evaluate_neighbors)
        
    
    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_sample, depth_min, depth_max, iter, grid=None, weight=None, view_weights=None):
        
        num_src_features = len(src_features)
        num_src_projs = len(src_projs)
        batch, feature_channel, height, width = ref_feature.size()
        device = ref_feature.get_device()
        
        num_depth = depth_sample.size()[1]
        assert num_src_features == num_src_projs, "Patchmatch Evaluation: Different number of images and projection matrices"
        if view_weights != None:
            assert num_src_features == view_weights.size()[1], "Patchmatch Evaluation: Different number of images and view weights"
        
        pixel_wise_weight_sum = 0
        
        ref_feature = ref_feature.view(batch, self.G, feature_channel//self.G, height, width)

        similarity_sum = 0
        
        if self.stage == 3 and view_weights == None:
            view_weights = []
            for src_feature, src_proj in zip(src_features, src_projs):
                
                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)
                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                # group-wise correlation
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # pixel-wise view weight
                view_weight = self.pixel_wise_net(similarity)
                view_weights.append(view_weight)
                
                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) #[B,1,1,H,W]
                else:
                    similarity_sum += similarity*view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                    
                del warped_feature, src_feature, src_proj, similarity, view_weight
            del src_features, src_projs
            view_weights = torch.cat(view_weights,dim=1) #[B,4,H,W], 4 is the number of source views
            # aggregated matching cost across all the source views
            similarity = similarity_sum.div_(pixel_wise_weight_sum)
            del ref_feature, pixel_wise_weight_sum, similarity_sum
            # adaptive spatial cost aggregation
            score = self.similarity_net(similarity, grid, weight)
            del similarity, grid, weight
            
            # apply softmax to get probability
            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)
            
            # depth regression: expectation
            depth_sample = torch.sum(depth_sample * score, dim = 1)

            return depth_sample, score, view_weights.detach()
        else:
            i=0
            for src_feature, src_proj in zip(src_features, src_projs):
                warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_sample)
                warped_feature = warped_feature.view(batch, self.G, feature_channel//self.G, num_depth, height, width)
                similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
                # reuse the pixel-wise view weight from first iteration of Patchmatch on stage 3
                view_weight = view_weights[:,i].unsqueeze(1) #[B,1,H,W]
                i=i+1
                
                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) #[B,1,1,H,W]
                else:
                    similarity_sum += similarity*view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                
                del warped_feature, src_feature, src_proj, similarity, view_weight
            del src_features, src_projs
                
            # [B, G, Ndepth, H, W]
            similarity = similarity_sum.div_(pixel_wise_weight_sum)
            
            del ref_feature, pixel_wise_weight_sum, similarity_sum
            
            score = self.similarity_net(similarity, grid, weight)
            del similarity, grid, weight

            softmax = nn.LogSoftmax(dim=1)
            score = softmax(score)
            score = torch.exp(score)
            

            if self.stage == 1 and iter == self.iterations: 
                # depth regression: inverse depth regression
                depth_index = torch.arange(0, num_depth, 1, device=device).view(1, num_depth, 1, 1)
                depth_index = torch.sum(depth_index * score, dim = 1)
                
                inverse_min_depth = 1.0 / depth_sample[:,-1,:,:]
                inverse_max_depth = 1.0 / depth_sample[:,0,:,:]
                depth_sample = inverse_max_depth + depth_index / (num_depth - 1) * \
                                            (inverse_min_depth - inverse_max_depth)
                depth_sample = 1.0 / depth_sample
                
                return depth_sample, score
            
            # depth regression: expectation
            else:
                depth_sample = torch.sum(depth_sample * score, dim = 1)
                
                return depth_sample, score
            


class PatchMatch(nn.Module):
    def __init__(self, random_initialization = False, propagation_out_range = 2, 
                patchmatch_iteration = 2, patchmatch_num_sample = 16, patchmatch_interval_scale = 0.025,
                num_feature = 64, G = 8, propagate_neighbors = 16, stage=3, evaluate_neighbors=9):
        super(PatchMatch, self).__init__()
        self.random_initialization = random_initialization
        self.depth_initialization = DepthInitialization(patchmatch_num_sample)
        self.propagation_out_range = propagation_out_range
        self.propagation = Propagation(propagate_neighbors)
        self.patchmatch_iteration = patchmatch_iteration
        
        self.patchmatch_interval_scale = patchmatch_interval_scale
        self.propa_num_feature = num_feature
        # group wise correlation
        self.G = G

        self.stage = stage
        
        self.dilation = propagation_out_range
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.evaluation = Evaluation(self.G, self.stage, self.evaluate_neighbors, self.patchmatch_iteration)
        # adaptive propagation
        if self.propagate_neighbors > 0:
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            if not (self.stage == 1 and self.patchmatch_iteration == 1):
                self.propa_conv = nn.Conv2d(
                                self.propa_num_feature,
                                2 * self.propagate_neighbors,
                                kernel_size=3,
                                stride=1,
                                padding=self.dilation,
                                dilation=self.dilation,
                                bias=True)
                nn.init.constant_(self.propa_conv.weight, 0.)
                nn.init.constant_(self.propa_conv.bias, 0.)

        # adaptive spatial cost aggregation (adaptive evaluation)
        self.eval_conv = nn.Conv2d(self.propa_num_feature, 2 * self.evaluate_neighbors, kernel_size=3, stride=1, 
                                    padding=self.dilation, dilation=self.dilation, bias=True)
        nn.init.constant_(self.eval_conv.weight, 0.)
        nn.init.constant_(self.eval_conv.bias, 0.)
        self.feature_weight_net = FeatureWeightNet(num_feature, self.evaluate_neighbors, self.G)
        

    # compute the offset for adaptive propagation
    def get_propagation_grid(self, batch, height, width, offset, device, img=None):
        if self.propagate_neighbors == 4:
            original_offset = [ [-self.dilation, 0],
                                [0,             -self.dilation],  [0,              self.dilation],
                                [self.dilation,  0]]
        elif self.propagate_neighbors == 8:
            original_offset = [[-self.dilation, -self.dilation], [-self.dilation, 0], [-self.dilation, self.dilation],
                                [0,             -self.dilation],  [0,              self.dilation],
                                [self.dilation, -self.dilation], [self.dilation,  0], [self.dilation,  self.dilation]]
        elif self.propagate_neighbors == 16:
            original_offset = [[-self.dilation, -self.dilation], [-self.dilation, 0], [-self.dilation, self.dilation],
                                [0,             -self.dilation],  [0,              self.dilation],
                                [self.dilation, -self.dilation], [self.dilation,  0], [self.dilation,  self.dilation]]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                original_offset.append([2*offset_x, 2*offset_y])
        else:
            raise NotImplementedError
        
        with torch.no_grad(): 
            y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                                torch.arange(0, width, dtype=torch.float32, device=device)])
            y_grid, x_grid = y_grid.contiguous(), x_grid.contiguous()
            y_grid, x_grid = y_grid.view(height * width), x_grid.view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list=[]
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            
            offset_x = original_offset_x + offset[:,2*i,:].unsqueeze(1)
            offset_y = original_offset_y + offset[:,2*i+1,:].unsqueeze(1)
            
            xy_list.append((xy+torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))
            
            
        xy = torch.cat(xy_list, dim=2)    # [B, 2, 9, H*W]
        
        
        del xy_list, x_grid, y_grid
        
        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized, y_normalized
        grid = grid.view(batch, self.propagate_neighbors * height, width, 2)
        return grid

    # compute the offests for adaptive spatial cost aggregation in adaptive evaluation
    def get_evaluation_grid(self, batch, height, width, offset, device, img=None):
        
        if self.evaluate_neighbors==9:
            dilation = self.dilation-1 #dilation of evaluation is a little smaller than propagation
            original_offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation],
                                [0,             -dilation], [0,              0], [0,              dilation],
                                [dilation, -dilation], [dilation,  0], [dilation,  dilation]]
        elif self.evaluate_neighbors==17:
            dilation = self.dilation-1
            original_offset = [[-dilation, -dilation], [-dilation, 0], [-dilation, dilation],
                                [0,             -dilation], [0,              0], [0,              dilation],
                                [dilation, -dilation], [dilation,  0], [dilation,  dilation]]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y !=0:
                    original_offset.append([2*offset_x, 2*offset_y])
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                                torch.arange(0, width, dtype=torch.float32, device=device)])
            y_grid, x_grid = y_grid.contiguous(), x_grid.contiguous()
            y_grid, x_grid = y_grid.view(height * width), x_grid.view(height * width)
            xy = torch.stack((x_grid, y_grid))  # [2, H*W]
            xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list=[]
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            
            offset_x = original_offset_x + offset[:,2*i,:].unsqueeze(1)
            offset_y = original_offset_y + offset[:,2*i+1,:].unsqueeze(1)
            
            xy_list.append((xy+torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))
            
            
        xy = torch.cat(xy_list, dim=2)    # [B, 2, 9, H*W]
        
        del xy_list, x_grid, y_grid
        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized, y_normalized
        grid = grid.view(batch, len(original_offset) * height, width, 2)
        return grid


    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_min, depth_max,
                depth = None, img = None, view_weights = None):
        depth_samples = []

        device = ref_feature.get_device()
        batch, _, height, width = ref_feature.size()
        
        # the learned additional 2D offsets for adaptive propagation
        if self.propagate_neighbors > 0:
            # last iteration on stage 1 does not have propagation (photometric consistency filtering)
            if not (self.stage == 1 and self.patchmatch_iteration == 1):
                propa_offset = self.propa_conv(ref_feature)
                propa_offset = propa_offset.view(batch, 2 * self.propagate_neighbors, height*width)
                propa_grid = self.get_propagation_grid(batch,height,width,propa_offset,device,img)
    
        # the learned additional 2D offsets for adaptive spatial cost aggregation (adaptive evaluation)
        eval_offset = self.eval_conv(ref_feature)
        eval_offset = eval_offset.view(batch, 2 * self.evaluate_neighbors, height*width)
        eval_grid = self.get_evaluation_grid(batch,height,width,eval_offset,device,img)

        feature_weight = self.feature_weight_net(ref_feature.detach(), eval_grid)
        
        
        # first iteration of Patchmatch
        iter = 1
        if self.random_initialization:
            # first iteration on stage 3, random initialization, no adaptive propagation
            depth_sample = self.depth_initialization(True, depth_min, depth_max, height, width, 
                                    self.patchmatch_interval_scale, device)
            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = depth_weight(depth_sample.detach(), depth_min, depth_max, eval_grid.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            
            # evaluation, outputs regressed depth map and pixel-wise view weights which will
            # be used for subsequent iterations
            depth_sample, score, view_weights = self.evaluation(ref_feature, src_features, ref_proj, src_projs, 
                                        depth_sample, depth_min, depth_max, iter, eval_grid, weight, view_weights)
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
        else:
            # subsequent iterations, local perturbation based on previous result
            depth_sample = self.depth_initialization(False, depth_min, depth_max, 
                                height, width, self.patchmatch_interval_scale, device, depth)
            del depth

            # adaptive propagation
            if self.propagate_neighbors > 0:
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                if not (self.stage == 1 and iter == self.patchmatch_iteration):
                    depth_sample = self.propagation(batch, height, width, depth_sample, propa_grid, depth_min, depth_max, 
                                            self.patchmatch_interval_scale)
            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = depth_weight(depth_sample.detach(), depth_min, depth_max, eval_grid.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            
            # evaluation, outputs regressed depth map
            depth_sample, score = self.evaluation(ref_feature, src_features, ref_proj, src_projs, 
                                        depth_sample, depth_min, depth_max, iter, eval_grid, weight, view_weights)
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)


        for iter in range(2, self.patchmatch_iteration+1):
            # local perturbation based on previous result
            depth_sample = self.depth_initialization(False, depth_min, depth_max, height, width, self.patchmatch_interval_scale, device, depth_sample)
            
            # adaptive propagation
            if self.propagate_neighbors > 0:
                # last iteration on stage 1 does not have propagation (photometric consistency filtering)
                if not (self.stage == 1 and iter == self.patchmatch_iteration):
                    depth_sample = self.propagation(batch, height, width, depth_sample, propa_grid, depth_min, depth_max, 
                                            self.patchmatch_interval_scale)
            # weights for adaptive spatial cost aggregation in adaptive evaluation
            weight = depth_weight(depth_sample.detach(), depth_min, depth_max, eval_grid.detach(), self.patchmatch_interval_scale,
                                    self.evaluate_neighbors)
            weight = weight * feature_weight.unsqueeze(1)
            weight = weight / torch.sum(weight, dim=2).unsqueeze(2)
            
            # evaluation, outputs regressed depth map
            depth_sample, score = self.evaluation(ref_feature, src_features, 
                                                ref_proj, src_projs, depth_sample, depth_min, depth_max, iter, eval_grid, weight, view_weights)
            
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
        
        return depth_samples, score, view_weights
        

# first, do convolution on aggregated cost among all the source views
# second, perform adaptive spatial cost aggregation to get final cost
class SimilarityNet(nn.Module):
    def __init__(self, G, neighbors = 9):
        super(SimilarityNet, self).__init__()
        self.neighbors = neighbors
        
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x1, grid, weight):
        # x1: [B, G, Ndepth, H, W], aggregated cost among all the source views with pixel-wise view weight
        # grid: position of sampling points in adaptive spatial cost aggregation
        # weight: weight of sampling points in adaptive spatial cost aggregation, combination of 
        # feature weight and depth weight
        
        batch,G,num_depth,height,width = x1.size() 
        
        x1 = self.similarity(self.conv1(self.conv0(x1))).squeeze(1)
        
        x1 = F.grid_sample(x1, 
                        grid, 
                        mode='bilinear',
                        padding_mode='border')
        
        # [B,Ndepth,9,H,W]
        x1 = x1.view(batch, num_depth, self.neighbors, height, width)
    
        return torch.sum(x1*weight, dim=2)
        

# adaptive spatial cost aggregation
# weight based on similarity of features of sampling points and center pixel
class FeatureWeightNet(nn.Module):
    def __init__(self, num_feature, neighbors=9, G=8):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = G

        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        
        self.output = nn.Sigmoid()

    def forward(self, ref_feature, grid):
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation
        batch,feature_channel,height,width = ref_feature.size()
        
        x = F.grid_sample(ref_feature, 
                        grid, 
                        mode='bilinear',
                        padding_mode='border')
        
        # [B,G,C//G,H,W]
        ref_feature = ref_feature.view(batch, self.G, feature_channel//self.G, height, width)

        x = x.view(batch, self.G, feature_channel//self.G, self.neighbors, height, width)
        # [B,G,Neighbor,H,W]
        x = (x * ref_feature.unsqueeze(3)).mean(2)
        del ref_feature
        # [B,Neighbor,H,W]
        x = self.similarity(self.conv1(self.conv0(x))).squeeze(1)
        
        
        return self.output(x)

# adaptive spatial cost aggregation
# weight based on depth difference of sampling points and center pixel
def depth_weight(depth_sample, depth_min, depth_max, grid, patchmatch_interval_scale, evaluate_neighbors):
    # grid: position of sampling points in adaptive spatial cost aggregation
    neighbors = evaluate_neighbors
    batch,num_depth,height,width = depth_sample.size()
    # normalization
    x = 1.0 / depth_sample
    del depth_sample
    inverse_depth_min = 1.0 / depth_min
    inverse_depth_max = 1.0 / depth_max
    x = (x-inverse_depth_max.view(batch,1,1,1))/(inverse_depth_min.view(batch,1,1,1)\
                    -inverse_depth_max.view(batch,1,1,1))
    
    x1 = F.grid_sample(x, 
                    grid, 
                    mode='bilinear',
                    padding_mode='border')
    del grid
    x1 = x1.view(batch, num_depth, neighbors, height, width)
    
    # [B,Ndepth,N_neighbors,H,W]
    x1 = torch.abs(x1 - x.unsqueeze(2)) / patchmatch_interval_scale
    del x
    x1 = torch.clamp(x1, min=0, max=4)
    # sigmoid output approximate to 1 when x=4
    x1 = (-x1 + 2) * 2
    output = nn.Sigmoid()
    x1 = output(x1)
    
    return x1.detach()

# estimate pixel-wise view weight
class PixelwiseNet(nn.Module):
    def __init__(self, G):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.conv2 = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
        

    def forward(self, x1):
        # x1: [B, G, Ndepth, H, W]
        
        # [B, Ndepth, H, W]
        x1 =self.conv2(self.conv1(self.conv0(x1))).squeeze(1)
        
        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]
        
        return output.unsqueeze(1)
        
