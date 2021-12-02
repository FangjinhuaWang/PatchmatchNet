import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import List
from datasets.dtu_yao import MVSDataset
from models import *
from utils import *
import sys
import datetime

cudnn.benchmark = True

parser = argparse.ArgumentParser(description="PatchmatchNet for high-resolution multi-view stereo")
parser.add_argument("--mode", default="train", help="train or val", choices=["train", "val"]) # done

parser.add_argument("--trainpath", help="train datapath") # done
parser.add_argument("--valpath", help="validation datapath") # not needed
parser.add_argument("--trainlist", help="train list") # done
parser.add_argument("--vallist", help="validation list") # done

parser.add_argument("--epochs", type=int, default=16, help="number of epochs to train") # done
parser.add_argument("--lr", type=float, default=0.001, help="learning rate") # done
parser.add_argument("--lrepochs", type=str, default="10,12,14:2",
                    help="epoch ids to downscale lr and the downscale rate") # done
parser.add_argument("--wd", type=float, default=0.0, help="weight decay") # done

parser.add_argument("--batch_size", type=int, default=12, help="train batch size") # done
parser.add_argument("--loadckpt", default=None, help="load a specific checkpoint") # done
parser.add_argument("--logdir", default="./checkpoints/debug", help="the directory to save checkpoints/logs") # done
parser.add_argument("--resume", action="store_true", help="continue to train the model") # done

parser.add_argument("--summary_freq", type=int, default=20, help="print and summary frequency") # done
parser.add_argument("--save_freq", type=int, default=1, help="save checkpoint frequency") # done
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed") # done
parser.add_argument("--parallel", action="store_true", default=False,
                    help="If set use DataParallel; this prevents TorchScript module export.") # done

parser.add_argument("--patchmatch_iteration", nargs="+", type=int, default=[1, 2, 2],
                    help="num of iteration of patchmatch on stages 1,2,3")
parser.add_argument("--patchmatch_num_sample", nargs="+", type=int, default=[8, 8, 16],
                    help="num of generated samples in local perturbation on stages 1,2,3")
parser.add_argument("--patchmatch_interval_scale", nargs="+", type=float, default=[0.005, 0.0125, 0.025], 
                    help="normalized interval in inverse depth range to generate samples in local perturbation")
parser.add_argument("--patchmatch_range", nargs="+", type=int, default=[6, 4, 2],
                    help="fixed offset of sampling points for propogation of patchmatch on stages 1,2,3")
parser.add_argument("--propagate_neighbors", nargs="+", type=int, default=[0, 8, 16],
                    help="num of neighbors for adaptive propagation on stages 1,2,3")
parser.add_argument("--evaluate_neighbors", nargs="+", type=int, default=[9, 9, 9],
                    help="num of neighbors for adaptive matching cost aggregation of adaptive evaluation on stages 1,2,3")


# parse arguments and check
args = parser.parse_args()
if args.resume:  # store_true means set the variable as "True"
    assert args.mode == "train"
    assert args.loadckpt is None
if args.valpath is None:
    args.valpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 5, robust_train=True)
test_dataset = MVSDataset(args.valpath, args.vallist, "val", 5,  robust_train=False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = PatchmatchNet(
    patchmatch_interval_scale=args.patchmatch_interval_scale,
    propagation_range=args.patchmatch_range,
    patchmatch_iteration=args.patchmatch_iteration,
    patchmatch_num_sample=args.patchmatch_num_sample,
    propagate_neighbors=args.propagate_neighbors,
    evaluate_neighbors=args.evaluate_neighbors
)
if args.parallel and args.mode in ["train", "val"]:
    model = nn.DataParallel(model)
model.cuda()
model_loss = patchmatchnet_loss
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)


# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    start_epoch = state_dict["epoch"] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict["model"])
print("start at epoch {}".format(start_epoch))
print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(":")[0].split(",")]
    lr_gamma = 1 / float(args.lrepochs.split(":")[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print("Epoch {}:".format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            do_summary_image = global_step % (50*args.summary_freq) == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, "train", scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, "train", image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                "Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}".format(
                    epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            if not args.parallel:
                model.eval()
                sm = torch.jit.script(model)
                sm.save(os.path.join(args.logdir, "module_{:0>6}.pt".format(epoch_idx)))
                model.train()

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            # do_summary_test = global_step % (10*args.summary_freq) == 0
            do_summary_image = global_step % (50*args.summary_freq) == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, "test", scalar_outputs, global_step)
            if do_summary_image:
                save_images(logger, "test", image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
                epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        save_scalars(logger, "fulltest", avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())


def test():
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, _ = test_sample(sample, detailed_summary=False)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print("Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
            batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        if batch_idx % 100 == 0:
            print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    print("final", avg_test_scalars)


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()
    
    sample_cuda = to_cuda(sample)
    depth_gt = create_stage_images(sample_cuda["depth"])
    mask = create_stage_images(sample_cuda["mask"])
    
    _, _, depth_patchmatch = model(
        sample_cuda["images"],
        sample_cuda["intrinsics"],
        sample_cuda["extrinsics"],
        sample_cuda["depth_min"],
        sample_cuda["depth_max"]
    )

    loss = model_loss(depth_patchmatch, depth_gt, mask)
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_gt_stage_0": depth_gt[0] * mask[0],
        "depth_refined_stage_0": depth_patchmatch[0][-1] * mask[0],
        "depth_patchmatch_stage_1": depth_patchmatch[1][-1] * mask[1],
        "depth_patchmatch_stage_2": depth_patchmatch[2][-1] * mask[2],
        "depth_patchmatch_stage_3": depth_patchmatch[3][-1] * mask[3],
        "ref_img": sample["images"][0]
    }
    if detailed_summary:
        image_outputs["errormap_refined_stage_0"] = (depth_patchmatch[0][-1] - depth_gt[0]).abs() * mask[0]
        image_outputs["errormap_patchmatch_stage_1"] = (depth_patchmatch[1][-1] - depth_gt[1]).abs() * mask[1]
        image_outputs["errormap_patchmatch_stage_2"] = (depth_patchmatch[2][-1] - depth_gt[2]).abs() * mask[2]
        image_outputs["errormap_patchmatch_stage_3"] = (depth_patchmatch[3][-1] - depth_gt[3]).abs() * mask[3]

    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = absolute_depth_error_metrics(
        depth_patchmatch[3][-1], depth_gt[3], mask[3] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = absolute_depth_error_metrics(
        depth_patchmatch[2][-1], depth_gt[2], mask[2] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = absolute_depth_error_metrics(
        depth_patchmatch[1][-1], depth_gt[1], mask[1] > 0.5)
    scalar_outputs["abs_depth_error_refined_stage_0"] = absolute_depth_error_metrics(
        depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 1.0)
    # threshold = 2mm
    scalar_outputs["thres2mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 2.0)
    # threshold = 4mm
    scalar_outputs["thres4mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 4.0)
    # threshold = 8mm
    scalar_outputs["thres8mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 8.0)
    
    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = to_cuda(sample)
    depth_gt = create_stage_images(sample_cuda["depth"])
    mask = create_stage_images(sample_cuda["mask"])
    
    _, _, depth_patchmatch = model(
        sample_cuda["images"],
        sample_cuda["intrinsics"],
        sample_cuda["extrinsics"],
        sample_cuda["depth_min"],
        sample_cuda["depth_max"]
    )

    loss = model_loss(depth_patchmatch, depth_gt, mask)
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_gt_stage_0": depth_gt[0] * mask[0],
        "depth_refined_stage_0": depth_patchmatch[0][-1] * mask[0],
        "depth_patchmatch_stage_1": depth_patchmatch[1][-1] * mask[1],
        "depth_patchmatch_stage_2": depth_patchmatch[2][-1] * mask[2],
        "depth_patchmatch_stage_3": depth_patchmatch[3][-1] * mask[3],
        "ref_img": sample["images"][0]
    }
    if detailed_summary:
        image_outputs["errormap_refined_stage_0"] = (depth_patchmatch[0][-1] - depth_gt[0]).abs() * mask[0]
        image_outputs["errormap_patchmatch_stage_1"] = (depth_patchmatch[1][-1] - depth_gt[1]).abs() * mask[1]
        image_outputs["errormap_patchmatch_stage_2"] = (depth_patchmatch[2][-1] - depth_gt[2]).abs() * mask[2]
        image_outputs["errormap_patchmatch_stage_3"] = (depth_patchmatch[3][-1] - depth_gt[3]).abs() * mask[3]

    scalar_outputs["abs_depth_error_patchmatch_stage_3"] = absolute_depth_error_metrics(
        depth_patchmatch[3][-1], depth_gt[3], mask[3] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_2"] = absolute_depth_error_metrics(
        depth_patchmatch[2][-1], depth_gt[2], mask[2] > 0.5)
    scalar_outputs["abs_depth_error_patchmatch_stage_1"] = absolute_depth_error_metrics(
        depth_patchmatch[1][-1], depth_gt[1], mask[1] > 0.5)
    scalar_outputs["abs_depth_error_refined_stage_0"] = absolute_depth_error_metrics(
        depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5)
    # threshold = 1mm
    scalar_outputs["thres1mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 1.0)
    # threshold = 2mm
    scalar_outputs["thres2mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 2.0)
    # threshold = 4mm
    scalar_outputs["thres4mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 4.0)
    # threshold = 8mm
    scalar_outputs["thres8mm_error"] = threshold_metrics(depth_patchmatch[0][-1], depth_gt[0], mask[0] > 0.5, 8.0)
    
    return tensor2float(loss), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def create_stage_images(image: torch.Tensor) -> List[torch.Tensor]:
    return [
        image,
        F.interpolate(image, scale_factor=0.5, mode="nearest"),
        F.interpolate(image, scale_factor=0.25, mode="nearest"),
        F.interpolate(image, scale_factor=0.125, mode="nearest")
    ]


if __name__ == "__main__":
    if args.mode == "train":
        train()
    elif args.mode == "val":
        test()
