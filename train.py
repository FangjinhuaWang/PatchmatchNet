import argparse
import datetime
import os
import sys
import time
import torch.backends.cudnn
import torch.nn.functional as F
import torch.nn.parallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List, Tuple

from datasets.mvs import MVSDataset
from models import PatchmatchNet, patchmatchnet_loss
from utils import *


# main training function
def train(
        args,
        model: torch.nn.Module,
        train_image_loader: DataLoader,
        test_image_loader: DataLoader,
        optimizer: Optimizer,
        start_epoch: int
) -> None:
    milestones = [int(epoch_idx) for epoch_idx in args.lr_epochs.split(":")[0].split(",")]
    gamma = 1 / float(args.lr_epochs.split(":")[1])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=start_epoch - 1)

    os.makedirs(args.output_folder, exist_ok=True)
    print("current time: ", str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    print("creating new summary file")
    logger = SummaryWriter(args.output_folder)

    for epoch_idx in range(start_epoch, args.epochs):
        print("Epoch {}:".format(epoch_idx + 1))
        scheduler.step()

        # training
        process_samples(args, train_sample, "train", logger, model, train_image_loader, optimizer, epoch_idx)
        logger.flush()

        # checkpoint and TorchScript module
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save(
                {"epoch": epoch_idx, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(args.output_folder, "params_{:0>6}.ckpt".format(epoch_idx))
            )
            # There is only one child here (PatchmatchNet module), but we have to use the iterator to access it
            for child_model in model.children():
                child_model.eval()
                sm = torch.jit.script(child_model)
                sm.save(os.path.join(args.output_folder, "module_{:0>6}.pt".format(epoch_idx)))
                child_model.train()

        # testing
        process_samples(args, test_sample, "test", logger, model, test_image_loader, optimizer, epoch_idx)
        logger.flush()

    logger.close()


# main validation function
def test(model: torch.nn.Module, image_loader: DataLoader) -> None:
    avg_test_scalars = DictAverageMeter()
    num_images = len(image_loader)
    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()
        loss, scalar_outputs, _ = test_sample(model, sample)
        avg_test_scalars.update(scalar_outputs)
        print("Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
            batch_idx + 1, num_images, loss, time.time() - start_time))
        if (batch_idx + 1) % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx + 1, num_images, avg_test_scalars.mean()))
    print("final", avg_test_scalars.mean())


def process_samples(
        args,
        sample_function,
        tag: str,
        logger: SummaryWriter,
        model: torch.nn.Module,
        image_loader: DataLoader,
        optimizer: Optimizer,
        epoch_idx: int
) -> None:
    num_images = len(image_loader)
    global_step = num_images * epoch_idx
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(image_loader):
        start_time = time.time()
        global_step = num_images * epoch_idx + batch_idx
        do_scalar_summary = global_step % args.summary_freq == 0
        do_image_summary = global_step % (50 * args.summary_freq) == 0
        loss, scalar_outputs, image_outputs = sample_function(model, sample, do_image_summary, optimizer)

        if do_scalar_summary:
            save_scalars(logger, tag, scalar_outputs, global_step)
        if do_image_summary:
            save_images(logger, tag, image_outputs, global_step)

        avg_test_scalars.update(scalar_outputs)
        print("Epoch {}/{}, Iter {}/{}, {} loss = {:.3f}, time = {:.3f}".format(
            epoch_idx + 1, args.epochs, batch_idx + 1, num_images, tag, loss, time.time() - start_time))

    print("End of processing {} samples.".format(tag))
    if tag == "test":
        save_scalars(logger, "full_test", avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())


def train_sample(
        model: PatchmatchNet, sample: Dict, image_summary: bool, optimizer: Optimizer
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    return process_sample(model, sample, True, image_summary, optimizer)


@make_nograd_func
def test_sample(
        model: PatchmatchNet, sample: Dict, image_summary: bool = False, optimizer: Optimizer = None
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    return process_sample(model, sample, False, image_summary, optimizer)


def process_sample(
        model: PatchmatchNet, sample: Dict, is_training: bool, image_summary: bool, optimizer: Optimizer
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    if is_training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    sample_cuda = to_cuda(sample)

    _, _, depths = model(
        sample_cuda["images"],
        sample_cuda["intrinsics"],
        sample_cuda["extrinsics"],
        sample_cuda["depth_min"],
        sample_cuda["depth_max"]
    )
    depth_gt = create_stage_images(sample_cuda["depth_gt"])
    mask = [m.bool() for m in create_stage_images(sample_cuda["mask"].float())]
    loss = patchmatchnet_loss(depths, depth_gt, mask)

    if is_training:
        loss.backward()
        optimizer.step()

    scalar_outputs = {"loss": loss}

    if image_summary:
        image_outputs = {"ref-image": sample["images"][0], "depth-gt": depth_gt[0] * mask[0]}
    else:
        image_outputs: Dict[str, torch.Tensor] = {}

    # iterate over stages
    for i in range(4):
        scalar_outputs[f"depth-error-stage-{i}"] = absolute_depth_error_metrics(depths[i][-1], depth_gt[i], mask[i])
        if image_summary:
            image_outputs[f"depth-stage-{i}"] = depths[i][-1] * mask[i]
            image_outputs[f"error-map-stage-{i}"] = (depths[i][-1] - depth_gt[i]).abs() * mask[i]

    # iterate over thresholds
    for t in [1, 2, 4, 8]:
        scalar_outputs[f"threshold-{t}mm-error"] = threshold_metrics(depths[0][-1], depth_gt[0], mask[0], float(t))

    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def create_stage_images(image: torch.Tensor) -> List[torch.Tensor]:
    return [
        image,
        F.interpolate(image, scale_factor=0.5, mode="nearest"),
        F.interpolate(image, scale_factor=0.25, mode="nearest"),
        F.interpolate(image, scale_factor=0.125, mode="nearest")
    ]


def find_latest_checkpoint(path: str) -> str:
    saved_models = [fn for fn in os.listdir(path) if fn.endswith(".ckpt")]
    if len(saved_models) == 0:
        return ""

    saved_models = sorted(saved_models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(path, saved_models[-1])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="PatchMatchNet for high-resolution multi-view stereo")
    parser.add_argument("--mode", type=str, default="train", help="Execution mode", choices=["train", "test"])
    parser.add_argument("--input_folder", type=str, help="input data path")
    parser.add_argument("--output_folder", type=str, default="", help="output path")
    parser.add_argument("--checkpoint_path", type=str, default="", help="load a specific checkpoint for parameters")

    # Dataset loading options
    parser.add_argument("--num_views", type=int, default=5,
                        help="total views for each patch-match problem including reference")
    parser.add_argument("--image_max_dim", type=int, default=640, help="max image dimension")
    parser.add_argument("--train_list", type=str, help="training scan list text file")
    parser.add_argument("--test_list", type=str, help="validation scan list text file")
    parser.add_argument("--num_light_idx", type=int, default=-1, help="Number of light indexes in source images")
    parser.add_argument("--batch_size", type=int, default=12, help="train batch size")

    # Training options
    parser.add_argument("--resume", action="store_true", default=False, help="continue to train the model")
    parser.add_argument("--epochs", type=int, default=16, help="number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_epochs", type=str, default="10,12,14:2",
                        help="epoch ids to downscale lr and the downscale rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--summary_freq", type=int, default=20, help="print and summary frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save checkpoint frequency")
    parser.add_argument("--rand_seed", type=int, default=1, metavar="S", help="random seed")

    # PatchMatchNet module options (only used when not loading from file)
    parser.add_argument("--patchmatch_interval_scale", nargs="+", type=float, default=[0.005, 0.0125, 0.025],
                        help="normalized interval in inverse depth range to generate samples in local perturbation")
    parser.add_argument("--propagation_range", nargs="+", type=int, default=[6, 4, 2],
                        help="fixed offset of sampling points for propagation of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_iteration", nargs="+", type=int, default=[1, 2, 2],
                        help="num of iteration of patch match on stages 1,2,3")
    parser.add_argument("--patchmatch_num_sample", nargs="+", type=int, default=[8, 8, 16],
                        help="num of generated samples in local perturbation on stages 1,2,3")
    parser.add_argument("--propagate_neighbors", nargs="+", type=int, default=[0, 8, 16],
                        help="num of neighbors for adaptive propagation on stages 1,2,3")
    parser.add_argument("--evaluate_neighbors", nargs="+", type=int, default=[9, 9, 9],
                        help="num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3")

    # parse arguments and check
    input_args = parser.parse_args()

    print("argv:", sys.argv[1:])
    print_args(input_args)

    if not os.path.isdir(input_args.input_folder):
        raise Exception("Invalid input folder: {}".format(input_args.input_folder))
    if not os.path.isfile(input_args.train_list):
        raise Exception("Invalid training scan list file: {}".format(input_args.train_list))
    if not os.path.isfile(input_args.test_list):
        raise Exception("Invalid validation scan list file: {}".format(input_args.test_list))

    if not input_args.output_folder:
        input_args.output_folder = input_args.input_folder

    torch.manual_seed(input_args.rand_seed)
    torch.cuda.manual_seed(input_args.rand_seed)

    train_dataset = MVSDataset(
        data_path=input_args.input_folder,
        num_views=input_args.num_views,
        max_dim=input_args.image_max_dim,
        scan_list=input_args.train_list,
        num_light_idx=input_args.num_light_idx
    )
    test_dataset = MVSDataset(
        data_path=input_args.input_folder,
        num_views=input_args.num_views,
        max_dim=input_args.image_max_dim,
        scan_list=input_args.test_list,
        num_light_idx=input_args.num_light_idx
    )

    train_loader = DataLoader(train_dataset, input_args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_dataset, input_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model, optimizer
    pmnet_model = PatchmatchNet(
        patchmatch_interval_scale=input_args.patchmatch_interval_scale,
        propagation_range=input_args.propagation_range,
        patchmatch_iteration=input_args.patchmatch_iteration,
        patchmatch_num_sample=input_args.patchmatch_num_sample,
        propagate_neighbors=input_args.propagate_neighbors,
        evaluate_neighbors=input_args.evaluate_neighbors
    )

    pmnet_model = torch.nn.DataParallel(pmnet_model)
    pmnet_model.cuda()
    pmnet_optimizer = torch.optim.Adam(pmnet_model.parameters(), lr=input_args.learning_rate, betas=(0.9, 0.999),
                                       weight_decay=input_args.weight_decay)

    # If no checkpoint is provided, then use the latest if it exists
    if not input_args.checkpoint_path:
        input_args.checkpoint_path = find_latest_checkpoint(input_args.output_folder)

    if input_args.mode == "train":
        epoch_start = 0
        if input_args.resume:
            if not os.path.isfile(input_args.checkpoint_path):
                raise Exception("Invalid checkpoint file: {}".format(input_args.checkpoint_path))

            # load training parameters
            print("Resume training from checkpoint: ", input_args.checkpoint_path)
            state_dict = torch.load(input_args.checkpoint_path)
            pmnet_model.load_state_dict(state_dict["model"])
            pmnet_optimizer.load_state_dict(state_dict["optimizer"])
            epoch_start = state_dict["epoch"] + 1

        print("Start training at epoch {}".format(epoch_start + 1))
        print("Number of model parameters: {}".format(sum([p.data.nelement() for p in pmnet_model.parameters()])))
        train(input_args, pmnet_model, train_loader, test_loader, pmnet_optimizer, epoch_start)
    elif input_args.mode == "test":
        if not os.path.isfile(input_args.checkpoint_path):
            raise Exception("Invalid checkpoint file: {}".format(input_args.checkpoint_path))

        print("Validation using checkpoint: ", input_args.checkpoint_path)
        state_dict = torch.load(input_args.checkpoint_path)
        pmnet_model.load_state_dict(state_dict["model"])

        test(pmnet_model, test_loader)
