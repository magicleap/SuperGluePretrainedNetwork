import argparse
import random
import numpy as np
from datasets.megadepth import MegaDepthPairsDataset
import os
import torch.multiprocessing
from tqdm import tqdm
from datetime import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter

from training.matches_generator import SuperPointMatchesGenerator
from models.superglue_v2_metric_learning import SuperGlue
from training.average_meter import AverageMeter
from training.train_utils import data_to_device

parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--superglue', type=str, help='SuperGlue weights',
    default='/home/ostap/projects/DepthGlue/models/weights/superglue_outdoor.pth'
)
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=3,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
         ' (Must be positive)')
parser.add_argument(
    '--gt_positive_threshold', type=int, default=5,
    help='Maximum reprohection error for 2 keypoints to be considered as a ground truth match in matching generator.'
         ' (Must be positive)')
parser.add_argument(
    '--gt_negative_threshold', type=int, default=15,
    help='Maximum reprohection error for 2 keypoints to be considered as a ground truth match in matching generator.'
         ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running training. If two numbers, '
         'resize to the exact dimensions, if one number, resize the max '
         'dimension, if -1, do not resize')

parser.add_argument(
    '--batch_size', type=int, default=4,
    help='batch_size')
parser.add_argument(
    '--num_workers', type=int, default=6,
    help='Number of dataset workers')
parser.add_argument(
    '--device', type=str, default='cuda:0',
    help='Device to train on')
parser.add_argument(
    '--data_path', type=str, default='/datasets/extra_space2/ostap/MegaDepth',
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--epoch', type=int, default=20,
    help='Number of epoches')
parser.add_argument(
    '--learning_rate', type=float, default=0.00001,
    help='Learning rate')
parser.add_argument(
    '--scheduler_step_size', type=int, default=1,
    help='Number of iterations after which decrease learning rate')
parser.add_argument(
    '--scheduler_gamma', type=float, default=0.999997,
    help='Scheduler lr multiplier')
parser.add_argument(
    '--log_path', type=str, default='/home/ostap/logs/superglue/pairs3d',
    help='Path to directory with experiments')
parser.add_argument(
    '--log_every_step', type=int, default=1000,
    help='Log train loss every number of steps')

# parser.add_argument(
#     '--triplet_margin', type=float, default=0.5,
#     help='Log train loss every number of steps')
parser.add_argument(
    '--lowe_margin', type=float, default=0.5,
    help='Log train loss every number of steps')

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    opt = parser.parse_args()
    print(opt)

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
            'gt_positive_threshold': opt.gt_positive_threshold,
            'gt_negative_threshold': opt.gt_negative_threshold

        },
        'superglue': {
            'weights': opt.superglue if opt.superglue != 'none' else None,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            # 'triplet_margin': opt.triplet_margin,
            'lowe_margin': opt.lowe_margin
        }
    }

    # create log path and save config
    experiment_name = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path = os.path.join(opt.log_path, experiment_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(opt), f)

    writer = SummaryWriter(log_path)

    with open('/home/ostap/projects/DepthGlue/assets/megadepth_train_scenes.txt') as f:
        train_scenes_list = f.readlines()
        train_scenes_list = [s.rstrip() for s in train_scenes_list]

    # with open('/home/ostap/projects/DepthGlue/assets/megadepth_validation_scenes.txt') as f:
    #     val_scenes_list = f.readlines()
    #     val_scenes_list = [s.rstrip() for s in val_scenes_list]

    # load training data
    train_ds = MegaDepthPairsDataset(
        root_path=opt.data_path,
        scenes_list=['0012'],
        target_size=opt.resize
    )

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )

    device = torch.device(opt.device)
    device = torch.device(opt.device)

    superpoint = SuperPointMatchesGenerator(config.get('superpoint', {})).eval().to(device)
    superglue = SuperGlue(config.get('superglue', {})).to(device)

    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=opt.scheduler_step_size,
        gamma=opt.scheduler_gamma
    )
    loss_meter, triplet_loss_meter = AverageMeter(), AverageMeter()

    iter_num = 0
    # start training
    for epoch in range(1, opt.epoch + 1):
        status_bar = tqdm(total=len(train_dl))
        status_bar.set_description(f"Epoch {epoch}, lr {list(map(lambda x: x['lr'], optimizer.param_groups))}")

        superglue.train()
        for data in train_dl:
            superglue.zero_grad()

            data = data_to_device(data, device)
            with torch.no_grad():
                data = superpoint(data)

            pred = superglue.training_step(data)
            if pred['skip_train_step']:  # image has no keypoint
                continue
            # process loss
            loss = pred['loss']
            triplet_loss = pred['triplet_loss']
            (loss + triplet_loss).backward()
            loss_meter.add_value(loss.item())
            triplet_loss_meter.add_value(triplet_loss.item())

            optimizer.step()
            scheduler.step()

            status_bar.update()
            status_bar.set_postfix(loss=loss.item(), triplet=triplet_loss.item())

            iter_num += 1
            if iter_num % opt.log_every_step == 0:
                report_loss_value = loss_meter.get_value(last_values=opt.log_every_step)
                report_triplet_loss_value = triplet_loss_meter.get_value(last_values=opt.log_every_step)

                print(f'loss: {report_loss_value}, triplet: {report_triplet_loss_value}')
                writer.add_scalar('Train Loss', report_loss_value, iter_num)
                writer.add_scalar('Train triplet Loss', report_triplet_loss_value, iter_num)

        writer.add_scalar('Train Loss (avg epoch)', loss_meter.get_value(), epoch)
        writer.add_scalar('Train triplet Loss (avg epoch)', triplet_loss_meter.get_value(), epoch)

        loss_meter.reset()
        triplet_loss_meter.reset()

        torch.save(superglue.state_dict(), os.path.join(log_path, f'superglue_outdoor_epoch_{epoch}.pth'))
