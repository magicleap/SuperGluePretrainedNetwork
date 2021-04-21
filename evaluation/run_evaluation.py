import argparse

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
        '--scenes_txt', type=str, default='/home/dobko/superglue_experiments/DepthGlue/assets/megadepth_validation_scenes.txt',
        help='Path to the file with scenes names')
    parser.add_argument(
        '--size_image', type=int, default=352,
        help='Target size')

    parser.add_argument(
        '--superglue', default='/home/ostap/projects/DepthGlue/models/weights/superglue_outdoor.pth',
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

    # Load dataset
    with open(opt.scenes_txt) as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    dataset = MegaDepthPairsDataset(
        root_path=opt.data_dir,
        scenes_list=scenes_list,
        target_size=(opt.size_image, opt.size_image)
    )

    # Initialize evaluator
    evaluator = Evaluator(model_config)

    # Perform evaluation on selected dataset
    results = evaluator.evaluate(dataset)
    print(results)
