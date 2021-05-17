import argparse
import os
import yaml
import tqdm
import torch
from evaluator import Evaluator
from datasets.megadepth import MegaDepthWarpingDataset, MegaDepthPairsDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_dir', type=str, default='/datasets/extra_space2/ostap/MegaDepth',
        help='Path to the directory of images')
    parser.add_argument(
        '--scenes_txt', type=str, default='/home/ostap/projects/DepthGlue/assets/new_megadepth_validation_scenes.txt',
        help='Path to the file with scenes names')
    parser.add_argument(
        '--size_image', type=int, default=[960, 720],
        help='Target size')

    parser.add_argument(
        '--superglue',
        default='/home/ostap/logs/superglue/pairs3d/2021-05-07-22-45-03/superglue_outdoor_iter_25000.pth',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    opt = parser.parse_args()
    print(opt)

    model_config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    logdir = os.path.abspath(os.path.join(opt.superglue, os.path.pardir))
    basename = os.path.basename(opt.superglue)

    # Load dataset
    with open(opt.scenes_txt) as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    dataset = MegaDepthPairsDataset(
        root_path=opt.data_dir,
        scenes_list=scenes_list,
        target_size=opt.size_image,
        # train=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        num_workers=4
    )
    print(f'Dataset length: {len(dataset)}')

    # Initialize evaluator
    evaluator = Evaluator(model_config, device='cuda')

    # Perform evaluation on selected dataset
    results = evaluator.evaluate(dataloader)
    print(results)
    with open(os.path.join(logdir, f'results_{basename[:-4]}.yaml'), 'w') as f:
        yaml.dump(results, f)
