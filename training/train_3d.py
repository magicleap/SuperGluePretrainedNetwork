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
from evaluation.evaluator import Evaluator

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
    '--gt_positive_threshold', type=int, default=3,
    help='Maximum reprohection error for 2 keypoints to be considered as a ground truth match in matching generator.'
         ' (Must be positive)')
parser.add_argument(
    '--gt_negative_threshold', type=int, default=5,
    help='Maximum reprohection error for 2 keypoints to be considered as a ground truth match in matching generator.'
         ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[960, 720],
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
    '--device', type=str, default='cuda',
    help='Device to train on')
parser.add_argument(
    '--data_path', type=str, default='/datasets/extra_space2/ostap/MegaDepth',
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--epoch', type=int, default=20,
    help='Number of epoches')
parser.add_argument(
    '--learning_rate', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument(
    '--grad_acum_steps', type=int, default=4,
    help='Number of iterations after which decrease learning rate')
parser.add_argument(
    '--scheduler_gamma', type=float, default=0.99997,
    help='Scheduler lr multiplier')
parser.add_argument(
    '--log_path', type=str, default='/home/ostap/logs/superglue/pairs3d',
    help='Path to directory with experiments')
parser.add_argument(
    '--log_every_step', type=int, default=3000,
    help='Log train loss every number of steps')
parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed')
parser.add_argument(
    '--triplet_margin', type=float, default=0.5,
    help='Log train loss every number of steps')

if __name__ == '__main__':

    opt = parser.parse_args()
    print(opt)

    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            'triplet_margin': opt.triplet_margin,
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

    with open('/home/ostap/projects/DepthGlue/assets/new_megadepth_validation_scenes.txt') as f:
        val_scenes_list = f.readlines()
        val_scenes_list = [s.rstrip() for s in val_scenes_list]

    # load training data
    train_ds = MegaDepthPairsDataset(
        root_path=opt.data_path,
        scenes_list=train_scenes_list,
        target_size=opt.resize,
        train=True
    )
    val_ds = MegaDepthPairsDataset(
        root_path=opt.data_path,
        scenes_list=val_scenes_list,
        target_size=opt.resize,
        train=False
    )
    print(len(train_ds), len(val_ds))

    train_dl = torch.utils.data.DataLoader(
        dataset=train_ds,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )

    val_dl = torch.utils.data.DataLoader(
        dataset=val_ds,
        shuffle=False,
        batch_size=1,
        num_workers=2
    )
    device = torch.device(opt.device)

    superpoint = SuperPointMatchesGenerator(config.get('superpoint', {})).eval().to(device)
    superglue = SuperGlue(config.get('superglue', {})).to(device)

    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=1,
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
            iter_num += 1
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

            # grad accumulation
            if iter_num % opt.grad_acum_steps == 0:
                optimizer.step()
                scheduler.step()
                superglue.zero_grad()

            status_bar.update()
            status_bar.set_postfix(loss=loss.item(), triplet=triplet_loss.item())


            if iter_num % opt.log_every_step == 0:
                report_loss_value = loss_meter.get_value(last_values=opt.log_every_step)
                report_triplet_loss_value = triplet_loss_meter.get_value(last_values=opt.log_every_step)

                print(f'loss: {report_loss_value}, triplet: {report_triplet_loss_value}')
                writer.add_scalar('Train Loss', report_loss_value, iter_num)
                writer.add_scalar('Train triplet Loss', report_triplet_loss_value, iter_num)
            if iter_num % (1 * opt.log_every_step) == 0:
                torch.save(superglue.state_dict(), os.path.join(log_path, f'superglue_outdoor_iter_{iter_num}.pth'))

                # validate here
                model_config = {
                    'superpoint': {
                        'nms_radius': 4,
                        'keypoint_threshold': 0.005,
                        'max_keypoints': 1024
                    },
                    'superglue': {
                        'weights': os.path.join(log_path, f'superglue_outdoor_iter_{iter_num}.pth'),
                        'sinkhorn_iterations': 20,
                        'match_threshold': 0.2,
                    }
                }
                val_evaluator = Evaluator(model_config, device=device)
                val_metrics  = val_evaluator.evaluate(val_dl)
                for k, v in val_metrics.items():
                    writer.add_scalar(f'Val {k}', v, iter_num)


        writer.add_scalar('Train Loss (avg epoch)', loss_meter.get_value(), epoch)
        writer.add_scalar('Train triplet Loss (avg epoch)', triplet_loss_meter.get_value(), epoch)

        loss_meter.reset()
        triplet_loss_meter.reset()

        torch.save(superglue.state_dict(), os.path.join(log_path, f'superglue_outdoor_epoch_{epoch}.pth'))
