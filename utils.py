import numpy as np
import torch
import torchvision.utils

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

# print arguments
def print_args(args):
    print('################################  args  ################################')
    for k, v in args.__dict__.items():
        print('{0: <10}\t{1: <30}\t{2: <20}'.format(k, str(v), str(type(v))))
    print('########################################################################')


# torch.no_grad wrapper for functions
def make_no_grad_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(args):
        if isinstance(args, list):
            return [wrapper(x) for x in args]
        elif isinstance(args, tuple):
            return tuple([wrapper(x) for x in args])
        elif isinstance(args, dict):
            return {k: wrapper(v) for k, v in args.items()}
        else:
            return func(args)

    return wrapper


@make_recursive_func
def tensor2float(args):
    if isinstance(args, float):
        return args
    elif isinstance(args, torch.Tensor):
        return args.data.item()
    else:
        raise NotImplementedError('invalid input type {} for tensor2float'.format(type(args)))


@make_recursive_func
def tensor2numpy(args):
    if isinstance(args, np.ndarray):
        return args
    elif isinstance(args, torch.Tensor):
        return args.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError('invalid input type {} for tensor2numpy'.format(type(args)))


@make_recursive_func
def to_cuda(args):
    if isinstance(args, torch.Tensor):
        return args.cuda()
    elif isinstance(args, str):
        return args
    else:
        raise NotImplementedError('invalid input type {} for to_cuda'.format(type(args)))


def save_scalars(logger: SummaryWriter, mode: str, scalars: Dict[str, float], global_step: int):
    for key, value in scalars.items():
        if not isinstance(value, float):
            raise NotImplementedError('invalid data {}: {}'.format(key, type(value)))
        name = '{}/{}'.format(mode, key)
        logger.add_scalar(name, value, global_step)


def save_images(logger: SummaryWriter, mode: str, images: Dict[str, np.ndarray], global_step: int):
    def preprocess(img_name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError('invalid img shape {}:{} in save_images'.format(img_name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return torchvision.utils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images.items():
        if not isinstance(value, np.ndarray):
            raise NotImplementedError('invalid data {}: {}'.format(key, type(value)))
        name = '{}/{}'.format(mode, key)
        logger.add_image(name, preprocess(name, value), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input: Dict[str, float]):
        self.count += 1
        if len(self.data) == 0:
            for key, value in new_input.items():
                if not isinstance(value, float):
                    raise NotImplementedError('invalid data {}: {}'.format(key, type(value)))
                self.data[key] = value
        else:
            for key, value in new_input.items():
                if not isinstance(value, float):
                    raise NotImplementedError('invalid data {}: {}'.format(key, type(value)))
                self.data[key] += value

    def mean(self) -> Dict[str, float]:
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est: Tensor, depth_gt: Tensor, mask: Tensor, *args) -> Tensor:
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_no_grad_func
@compute_metrics_for_each_image
def threshold_metrics(depth_est: Tensor, depth_gt: Tensor, mask: Tensor, threshold: float) -> Tensor:
    # if threshold is int or float, then True
    assert isinstance(threshold, float)
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > threshold
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_no_grad_func
@compute_metrics_for_each_image
def abs_depth_error_metrics(depth_est: Tensor, depth_gt: Tensor, mask: Tensor) -> Tensor:
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())
