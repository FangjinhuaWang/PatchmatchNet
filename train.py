import argparse
import datetime
import os
import sys
import time
import torch.backends.cudnn
import torch.nn.parallel

from datasets.mvs import MVSDataset
from models import PatchMatchNet, patch_match_net_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import DictAverageMeter, abs_depth_error_metrics, make_no_grad_func, print_args, save_scalars, save_images,\
    tensor2float, to_cuda, threshold_metrics

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PatchMatchNet for high-resolution multi-view stereo')
parser.add_argument('--mode', type=str, default='train', help='Execution mode', choices=['train', 'test'])
parser.add_argument('--input_folder', type=str, help='input data path')
parser.add_argument('--output_folder', type=str, help='output path')
parser.add_argument('--checkpoint_path', type=str, help='load a specific checkpoint for parameters')
parser.add_argument('--data_parallel', type=bool, default=False, help='Flag to use or skip data parallel mode')

# Dataset loading options
parser.add_argument('--dataset_type', type=str, default='custom', help='Type of dataset to configure parameters',
                    choices=['custom', 'dtu', 'eth3d', 'blended'])
parser.add_argument('--num_views', type=int, default=11, help='total views for each patch-match problem including reference')
parser.add_argument('--image_max_dim', type=int, default=640, help='max image dimension')
parser.add_argument('--train_list', type=str, default='', help='Optional scan list text file to identify input folders')
parser.add_argument('--test_list', type=str, default='', help='Optional scan list text file to identify input folders')
parser.add_argument('--image_light_idx', type=int, default=-1, help='Image light indexes')
parser.add_argument('--batch_size', type=int, default=12, help='train batch size')

# Training options
parser.add_argument('--resume', type=bool, default=True, help='continue to train the model')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_epochs', type=str, default='10,12,14:2',
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--rand_seed', type=int, default=1, metavar='S', help='random seed')

# PatchMatchNet module options (only used when not loading from file)
parser.add_argument('--patch_match_interval_scale', nargs='+', type=float, default=[0.005, 0.0125, 0.025],
                    help='normalized interval in inverse depth range to generate samples in local perturbation')
parser.add_argument('--propagation_range', nargs='+', type=int, default=[6, 4, 2],
                    help='fixed offset of sampling points for propagation of patch match on stages 1,2,3')
parser.add_argument('--patch_match_iteration', nargs='+', type=int, default=[1, 2, 2],
                    help='num of iteration of patch match on stages 1,2,3')
parser.add_argument('--patch_match_num_sample', nargs='+', type=int, default=[8, 8, 16],
                    help='num of generated samples in local perturbation on stages 1,2,3')
parser.add_argument('--propagate_neighbors', nargs='+', type=int, default=[0, 8, 16],
                    help='num of neighbors for adaptive propagation on stages 1,2,3')
parser.add_argument('--evaluate_neighbors', nargs='+', type=int, default=[9, 9, 9],
                    help='num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3')

# parse arguments and check
args = parser.parse_args()
print('argv:', sys.argv[1:])
print_args(args)

torch.manual_seed(args.rand_seed)
torch.cuda.manual_seed(args.rand_seed)

if args.mode == 'train':
    os.makedirs(args.output_folder, exist_ok=True)
    print('current time', str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    print('creating new summary file')
    logger = SummaryWriter(args.output_folder)

# dataset and loader setup
num_light_idx = -1
cam_folder = 'cams'
pair_path = 'pair.txt'
image_folder = 'images'
depth_folder = 'depth_gt'
index_path = None
if args.dataset_type == 'dtu':
    num_light_idx = 7
elif args.dataset_type == 'eth3d':
    pair_path = 'cams/pair.txt'
    depth_folder = 'depths'
    index_path = 'cams/index2prefix.txt'
elif args.dataset_type == 'blended':
    pair_path = 'cams/pair.txt'
    image_folder = 'blended_images'
    depth_folder = 'rendered_depth_maps'

dataset = MVSDataset(args.input_folder, args.num_views, args.image_max_dim, args.scan_list, False, num_light_idx,
                     cam_folder, pair_path, image_folder, depth_folder, index_path)

train_dataset = MVSDataset(args.input_folder, args.num_views, args.image_max_dim, args.train_list, True, num_light_idx,
                           cam_folder, pair_path, image_folder, depth_folder, index_path)
test_dataset = MVSDataset(args.input_folder, args.num_views, args.image_max_dim, args.test_list, False, num_light_idx,
                          cam_folder, pair_path, image_folder, depth_folder, index_path)

train_image_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
test_image_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = PatchMatchNet(args.patch_match_interval_scale, args.propagation_range, args.patch_match_iteration,
                      args.patch_match_num_sample, args.propagate_neighbors, args.evaluate_neighbors, True)

if args.data_parallel:
    model = torch.nn.DataParallel(model)

model.cuda()
model_loss = patch_match_net_loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                             weight_decay=args.weight_decay)

# load parameters
start_epoch = 0
if args.resume:
    if args.checkpoint_path:
        print('Loading specific checkpoint: ', args.checkpoint_path)
    else:
        saved_models = [fn for fn in os.listdir(args.output_folder) if fn.endswith('.pt')]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        args.checkpoint_path = os.path.join(args.output_folder, saved_models[-1])
        print('Loading latest checkpoint: ', args.checkpoint_path)

    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1

print('Start at epoch {}'.format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lr_epochs.split(':')[0].split(',')]
    gamma = 1 / float(args.lr_epochs.split(':')[1])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        scheduler.step()

        # training
        process_samples(train_image_loader, epoch_idx, True)

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                '{}/model_{:0>6}.pt'.format(args.output_folder, epoch_idx))

        # testing
        process_samples(test_image_loader, epoch_idx, False)


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(test_image_loader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(test_image_loader), loss,
                                                                    time.time() - start_time))
        if batch_idx % 100 == 0:
            print(
                'Iter {}/{}, test results = {}'.format(batch_idx, len(test_image_loader), avg_test_scalars.mean()))
    print('final', avg_test_scalars)


def process_samples(image_loader, epoch_idx: int, train_flag: bool):
    num_images = len(image_loader)
    global_step = num_images * epoch_idx
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()
        global_step = num_images * epoch_idx + batch_idx
        do_summary = global_step % args.summary_freq == 0
        do_summary_image = global_step % (50 * args.summary_freq) == 0
        if train_flag:
            tag = 'train'
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
        else:
            tag = 'test'
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
        if do_summary:
            save_scalars(logger, tag, scalar_outputs, global_step)
        if do_summary_image:
            save_images(logger, tag, image_outputs, global_step)
        avg_test_scalars.update(scalar_outputs)
        print(
            'Epoch {}/{}, Iter {}/{}, {} loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, num_images,
                                                                              tag, loss, time.time() - start_time))
    if not train_flag:
        save_scalars(logger, 'full_test', avg_test_scalars.mean(), global_step)
        print('avg_test_scalars:', avg_test_scalars.mean())


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = to_cuda(sample)
    depth_gt = sample_cuda['depth_gt']
    mask = sample_cuda['mask']

    depth, _, depths = model.forward(sample_cuda['images'], sample_cuda['intrinsics'], sample_cuda['extrinsics'],
                                     sample_cuda['depth_params'])

    depth_est = outputs['refined_depth']

    depth_patchmatch = outputs['depth_patchmatch']

    loss = model_loss(depths, depth_gt, mask)
    loss.backward()
    optimizer.step()

    scalar_outputs = {'loss': loss}
    image_outputs = {'depth_refined_stage_0': depths[0] * mask,
                     'depth_gt_stage_0': depth_gt * mask,
                     'depth_patchmatch_stage_1': depth_patchmatch['stage_1'][-1] * mask['stage_1'],
                     'depth_patchmatch_stage_2': depth_patchmatch['stage_2'][-1] * mask['stage_2'],
                     'depth_patchmatch_stage_3': depth_patchmatch['stage_3'][-1] * mask['stage_3'],
                     'ref_img': sample['imgs']['stage_0'][:, 0],
                     }
    if detailed_summary:
        image_outputs['errormap_refined_stage_0'] = (depth_est['stage_0'] - depth_gt['stage_0']).abs() * mask['stage_0']
        image_outputs['errormap_patchmatch_stage_1'] = \
            (depth_patchmatch['stage_1'][-1] - depth_gt['stage_1']).abs() * mask['stage_1']
        image_outputs['errormap_patchmatch_stage_2'] = \
            (depth_patchmatch['stage_2'][-1] - depth_gt['stage_2']).abs() * mask['stage_2']
        image_outputs['errormap_patchmatch_stage_3'] = \
            (depth_patchmatch['stage_3'][-1] - depth_gt['stage_3']).abs() * mask['stage_3']

    scalar_outputs['abs_depth_error_refined_stage_0'] = abs_depth_error_metrics(depth_est['stage_0'],
                                                                                depth_gt['stage_0'],
                                                                                mask['stage_0'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_3'] = abs_depth_error_metrics(depth_patchmatch['stage_3'][-1],
                                                                                   depth_gt['stage_3'],
                                                                                   mask['stage_3'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_2'] = abs_depth_error_metrics(depth_patchmatch['stage_2'][-1],
                                                                                   depth_gt['stage_2'],
                                                                                   mask['stage_2'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_1'] = abs_depth_error_metrics(depth_patchmatch['stage_1'][-1],
                                                                                   depth_gt['stage_1'],
                                                                                   mask['stage_1'] > 0.5)
    # threshold = 1mm
    scalar_outputs['thres1mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 1)
    # threshold = 2mm
    scalar_outputs['thres2mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 2)
    # threshold = 4mm
    scalar_outputs['thres4mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 4)
    # threshold = 8mm
    scalar_outputs['thres8mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_no_grad_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = to_cuda(sample)
    depth_gt = sample_cuda['depth']
    mask = sample_cuda['mask']

    outputs = model(sample_cuda['imgs'], sample_cuda['proj_matrices'],
                    sample_cuda['depth_min'], sample_cuda['depth_max'])

    depth_est = outputs['refined_depth']
    depth_patchmatch = outputs['depth_patchmatch']

    loss = model_loss(depth_patchmatch, depth_est, depth_gt, mask)
    scalar_outputs = {'loss': loss}
    image_outputs = {'depth_refined_stage_0': depth_est['stage_0'] * mask['stage_0'],
                     'depth_gt_stage_0': depth_gt['stage_0'] * mask['stage_0'],
                     'depth_patchmatch_stage_1': depth_patchmatch['stage_1'][-1] * mask['stage_1'],
                     'depth_patchmatch_stage_2': depth_patchmatch['stage_2'][-1] * mask['stage_2'],
                     'depth_patchmatch_stage_3': depth_patchmatch['stage_3'][-1] * mask['stage_3'],
                     'ref_img': sample['imgs']['stage_0'][:, 0],
                     }
    if detailed_summary:
        image_outputs['errormap_refined_stage_0'] = (depth_est['stage_0'] - depth_gt['stage_0']).abs() * mask['stage_0']
        image_outputs['errormap_patchmatch_stage_1'] = \
            (depth_patchmatch['stage_1'][-1] - depth_gt['stage_1']).abs() * mask['stage_1']
        image_outputs['errormap_patchmatch_stage_2'] = \
            (depth_patchmatch['stage_2'][-1] - depth_gt['stage_2']).abs() * mask['stage_2']
        image_outputs['errormap_patchmatch_stage_3'] = \
            (depth_patchmatch['stage_3'][-1] - depth_gt['stage_3']).abs() * mask['stage_3']

    scalar_outputs['abs_depth_error_refined_stage_0'] = abs_depth_error_metrics(depth_est['stage_0'],
                                                                                depth_gt['stage_0'],
                                                                                mask['stage_0'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_3'] = abs_depth_error_metrics(depth_patchmatch['stage_3'][-1],
                                                                                   depth_gt['stage_3'],
                                                                                   mask['stage_3'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_2'] = abs_depth_error_metrics(depth_patchmatch['stage_2'][-1],
                                                                                   depth_gt['stage_2'],
                                                                                   mask['stage_2'] > 0.5)
    scalar_outputs['abs_depth_error_patchmatch_stage_1'] = abs_depth_error_metrics(depth_patchmatch['stage_1'][-1],
                                                                                   depth_gt['stage_1'],
                                                                                   mask['stage_1'] > 0.5)
    # threshold = 1mm
    scalar_outputs['thres1mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 1)
    # threshold = 2mm
    scalar_outputs['thres2mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 2)
    # threshold = 4mm
    scalar_outputs['thres4mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 4)
    # threshold = 8mm
    scalar_outputs['thres8mm_error'] = threshold_metrics(depth_est['stage_0'], depth_gt['stage_0'],
                                                         mask['stage_0'] > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
