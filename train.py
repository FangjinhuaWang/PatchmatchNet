import argparse
import datetime
import os
import sys
import time
import torch.backends.cudnn
import torch.nn.parallel

from datasets.mvs import MVSDataset
from models import PatchMatchNet, patch_match_net_loss
from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple
from utils import DictAverageMeter, abs_depth_error_metrics, make_no_grad_func, print_args, save_scalars, save_images,\
    tensor2float, tensor2numpy, to_cuda, threshold_metrics


# main function
def train(args, model: PatchMatchNet, train_image_loader: DataLoader, test_image_loader: DataLoader,
          optimizer: Optimizer, start_epoch: int):
    milestones = [int(epoch_idx) for epoch_idx in args.lr_epochs.split(':')[0].split(',')]
    gamma = 1 / float(args.lr_epochs.split(':')[1])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=start_epoch - 1)

    os.makedirs(args.output_folder, exist_ok=True)
    print('current time', str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    print('creating new summary file')
    logger = SummaryWriter(args.output_folder)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx + 1))
        scheduler.step()

        # training
        process_samples(args, logger, model, train_image_loader, optimizer, epoch_idx, True)
        logger.flush()

        # checkpoint and TorchScript module
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                os.path.join(args.output_folder, 'params_{:0>6}.pt'.format(epoch_idx)))

            model.eval()
            model.save_all_depths = False
            sm = torch.jit.script(model)
            sm.save(os.path.join(args.output_folder, 'module_{:0>6}.pt'.format(epoch_idx)))
            model.train()
            model.save_all_depths = True

        # testing
        process_samples(args, logger, model, test_image_loader, optimizer, epoch_idx, False)
        logger.flush()

    logger.close()


def test(model: PatchMatchNet, image_loader: DataLoader):
    avg_test_scalars = DictAverageMeter()
    num_images = len(image_loader)
    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()
        loss, scalar_outputs, _ = test_sample(model, sample)
        avg_test_scalars.update(scalar_outputs)
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx + 1, num_images, loss, time.time() - start_time))
        if (batch_idx + 1) % 100 == 0:
            print('Iter {}/{}, test results = {}'.format(batch_idx + 1, num_images, avg_test_scalars.mean()))
    print('final', avg_test_scalars.mean())


def process_samples(args, logger: SummaryWriter, model: PatchMatchNet, image_loader: DataLoader, optimizer: Optimizer,
                    epoch_idx: int, is_training: bool):
    num_images = len(image_loader)
    global_step = num_images * epoch_idx
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()
        global_step = num_images * epoch_idx + batch_idx
        do_scalar_summary = global_step % args.summary_freq == 0
        do_image_summary = global_step % (10 * args.summary_freq) == 0
        if is_training:
            tag = 'train'
            loss, scalar_outputs, image_outputs = train_sample(model, sample, optimizer, do_image_summary)
        else:
            tag = 'test'
            loss, scalar_outputs, image_outputs = test_sample(model, sample, do_image_summary)

        if do_scalar_summary:
            save_scalars(logger, tag, scalar_outputs, global_step)
        if do_image_summary:
            save_images(logger, tag, image_outputs, global_step)

        avg_test_scalars.update(scalar_outputs)
        print(
            'Epoch {}/{}, Iter {}/{}, {} loss = {:.3f}, time = {:.3f}'.format(epoch_idx + 1, args.epochs, batch_idx + 1,
                                                                              num_images, tag, loss, time.time() - start_time))
    if not is_training:
        save_scalars(logger, 'full_test', avg_test_scalars.mean(), global_step)
        print('avg_test_scalars:', avg_test_scalars.mean())


def train_sample(model: PatchMatchNet, sample: Dict, optimizer: Optimizer, image_summary: bool = False) \
        -> Tuple[float, Dict[str, float], Dict[str, ndarray]]:
    return process_sample(model, sample, True, image_summary, optimizer)


@make_no_grad_func
def test_sample(model: PatchMatchNet, sample: Dict, image_summary: bool = False) \
        -> Tuple[float, Dict[str, float], Dict[str, ndarray]]:
    return process_sample(model, sample, False, image_summary)


def process_sample(model: PatchMatchNet, sample: Dict, is_training: bool = True, image_summary: bool = False,
                   optimizer: Optimizer = None) -> Tuple[float, Dict[str, float], Dict[str, ndarray]]:
    if is_training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    sample_cuda = to_cuda(sample)
    depth_gt = sample_cuda['depth_gt']
    mask = sample_cuda['mask']

    _, _, depths = model.forward(sample_cuda['images'], sample_cuda['intrinsics'], sample_cuda['extrinsics'], sample_cuda['depth_params'])

    loss, gt_depths, masks = patch_match_net_loss(depths, depth_gt, mask)

    if is_training:
        loss.backward()
        optimizer.step()

    scalar_outputs = {'loss': loss,
                      'depth-error-stage-0': abs_depth_error_metrics(depths[0][-1], gt_depths[0], masks[0]),
                      'depth-error-stage-1': abs_depth_error_metrics(depths[1][-1], gt_depths[1], masks[1]),
                      'depth-error-stage-2': abs_depth_error_metrics(depths[2][-1], gt_depths[2], masks[2]),
                      'depth-error-stage-3': abs_depth_error_metrics(depths[3][-1], gt_depths[3], masks[3]),
                      'threshold-error-1mm': threshold_metrics(depths[0][-1], gt_depths[0], masks[0], 1.0),
                      'threshold-error-2mm': threshold_metrics(depths[0][-1], gt_depths[0], masks[0], 2.0),
                      'threshold-error-4mm': threshold_metrics(depths[0][-1], gt_depths[0], masks[0], 4.0),
                      'threshold-error-8mm': threshold_metrics(depths[0][-1], gt_depths[0], masks[0], 8.0)
                      }

    if image_summary:
        image_outputs = {'ref-image': sample['images'][0],
                         'depth-gt': gt_depths[0] * masks[0],
                         'depth-stage-0': depths[0][-1] * masks[0],
                         'depth-stage-1': depths[1][-1] * masks[1],
                         'depth-stage-2': depths[2][-1] * masks[2],
                         'depth-stage-3': depths[3][-1] * masks[3],
                         'error-map-stage-0': (depths[0][-1] - gt_depths[0]).abs() * masks[0],
                         'error-map-stage-1': (depths[1][-1] - gt_depths[1]).abs() * masks[1],
                         'error-map-stage-2': (depths[2][-1] - gt_depths[2]).abs() * masks[2],
                         'error-map-stage-3': (depths[3][-1] - gt_depths[3]).abs() * masks[3]
                         }
    else:
        image_outputs: Dict[str, Tensor] = {}

    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
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
    parser.add_argument('--num_views', type=int, default=5,
                        help='total views for each patch-match problem including reference')
    parser.add_argument('--image_max_dim', type=int, default=640, help='max image dimension')
    parser.add_argument('--train_list', type=str, default='',
                        help='Optional scan list text file to identify input folders')
    parser.add_argument('--test_list', type=str, default='',
                        help='Optional scan list text file to identify input folders')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

    # Training options
    parser.add_argument('--resume', type=bool, default=False, help='continue to train the model')
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
    input_args = parser.parse_args()
    # args.input_folder = r'C:\Users\anmatako\Downloads\datasets\dtu_train\scan1'
    # args.output_folder = r'C:\Users\anmatako\Downloads\test_outputs\dtu_train'

    print('argv:', sys.argv[1:])
    print_args(input_args)

    torch.manual_seed(input_args.rand_seed)
    torch.cuda.manual_seed(input_args.rand_seed)

    # dataset and loader setup
    num_light_idx = -1
    cam_folder = 'cams'
    pair_path = 'pair.txt'
    image_folder = 'images'
    depth_folder = 'depth_gt'
    index_path = None
    if input_args.dataset_type == 'dtu':
        num_light_idx = 7
    elif input_args.dataset_type == 'eth3d':
        pair_path = 'cams/pair.txt'
        depth_folder = 'depths'
        index_path = 'cams/index2prefix.txt'
    elif input_args.dataset_type == 'blended':
        pair_path = 'cams/pair.txt'
        image_folder = 'blended_images'
        depth_folder = 'rendered_depth_maps'

    train_dataset = MVSDataset(input_args.input_folder, input_args.num_views, input_args.image_max_dim, input_args.train_list, True,
                               num_light_idx,
                               cam_folder, pair_path, image_folder, depth_folder, index_path)
    test_dataset = MVSDataset(input_args.input_folder, input_args.num_views, input_args.image_max_dim, input_args.test_list, False,
                              num_light_idx,
                              cam_folder, pair_path, image_folder, depth_folder, index_path)

    train_loader = DataLoader(train_dataset, input_args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, input_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model, optimizer
    pmnet_model = PatchMatchNet(input_args.patch_match_interval_scale, input_args.propagation_range, input_args.patch_match_iteration,
                                input_args.patch_match_num_sample, input_args.propagate_neighbors, input_args.evaluate_neighbors, True)

    if input_args.data_parallel:
        pmnet_model = torch.nn.DataParallel(pmnet_model)

    pmnet_model.cuda()
    pmnet_optimizer = torch.optim.Adam(pmnet_model.parameters(), lr=input_args.learning_rate, betas=(0.9, 0.999),
                                       weight_decay=input_args.weight_decay)

    # load parameters
    epoch_start = 0
    if input_args.resume:
        if input_args.checkpoint_path:
            print('Loading specific checkpoint: ', input_args.checkpoint_path)
        else:
            saved_models = [fn for fn in os.listdir(input_args.output_folder) if
                            fn.startswith('params') and fn.endswith('.pt')]
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            input_args.checkpoint_path = os.path.join(input_args.output_folder, saved_models[-1])
            print('Loading latest checkpoint: ', input_args.checkpoint_path)

        state_dict = torch.load(input_args.checkpoint_path)
        pmnet_model.load_state_dict(state_dict['model'])
        pmnet_optimizer.load_state_dict(state_dict['optimizer'])
        epoch_start = state_dict['epoch'] + 1

    print('Start at epoch {}'.format(epoch_start))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in pmnet_model.parameters()])))

    if input_args.mode == 'train':
        train(input_args, pmnet_model, train_loader, test_loader, pmnet_optimizer, epoch_start)
    elif input_args.mode == 'test':
        test(pmnet_model, test_loader)
