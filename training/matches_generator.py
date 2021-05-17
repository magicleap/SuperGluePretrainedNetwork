import torch
import torch.nn as nn

from models.superpoint import SuperPoint
from training.train_utils import data_to_device, min_stack, reproject_keypoints, get_inverse_transformation


class SuperPointMatchesGenerator(nn.Module):
    """
    Class for generating ground truth matches using image pair and transformation between them.
    Keypoints are detected and described using SuperPoint model.
    """
    UNMATCHED_INDEX = -1 # index of keypoint that don't have a match
    IGNORE_INDEX = -2  # index of keypoints to ignore during loss calculation

    def __init__(self, config):
        super(SuperPointMatchesGenerator, self).__init__()
        self.superpoint = SuperPoint(config)
        self.gt_positive_threshold = config['gt_positive_threshold']
        self.gt_negative_threshold = config.get('gt_negative_threshold', self.gt_positive_threshold)

    def forward(self, data):
        pred0 = self.superpoint({'image': data['image0']})
        pred1 = self.superpoint({'image': data['image1']})

        pred0, pred1 = min_stack(pred0), min_stack(pred1)

        kpts0, kpts1 = pred0['keypoints'], pred1['keypoints']
        desc0, desc1 = pred0['descriptors'], pred1['descriptors']
        scores0, scores1 = pred0['scores'], pred1['scores']
        transformation = data['transformation']
        transformation_inv = get_inverse_transformation(transformation)
        num0, num1 = kpts0.size(1), kpts1.size(1)

        # establish ground truth correspondences given transformation
        kpts0_transformed, mask0 = reproject_keypoints(kpts0, transformation)
        kpts1_transformed, mask1 = reproject_keypoints(kpts1, transformation_inv)
        reprojection_error_0_to_1 = torch.cdist(kpts0_transformed, kpts1, p=2)  # batch_size x num0 x num1
        reprojection_error_1_to_0 = torch.cdist(kpts1_transformed, kpts0, p=2)  # batch_size x num1 x num0

        min_dist0, nn_matches0 = reprojection_error_0_to_1.min(2)  # batch_size x num0
        min_dist1, nn_matches1 = reprojection_error_1_to_0.min(2)  # batch_size x num1
        gt_matches0, gt_matches1 = nn_matches0.clone(), nn_matches1.clone()
        device = gt_matches0.device
        cross_check_consistent0 = torch.arange(num0, device=device).unsqueeze(0) == gt_matches1.gather(1, gt_matches0)
        gt_matches0[~cross_check_consistent0] = self.UNMATCHED_INDEX

        cross_check_consistent1 = torch.arange(num1, device=device).unsqueeze(0) == gt_matches0.gather(1, gt_matches1)
        gt_matches1[~cross_check_consistent1] = self.UNMATCHED_INDEX

        symmetric_dist = 0.5 * (min_dist0[cross_check_consistent0] + min_dist1[cross_check_consistent1])

        gt_matches0[cross_check_consistent0][symmetric_dist > self.gt_positive_threshold] = self.IGNORE_INDEX
        gt_matches0[cross_check_consistent0][symmetric_dist > self.gt_negative_threshold] = self.UNMATCHED_INDEX

        gt_matches1[cross_check_consistent1][symmetric_dist > self.gt_positive_threshold] = self.IGNORE_INDEX
        gt_matches1[cross_check_consistent1][symmetric_dist > self.gt_negative_threshold] = self.UNMATCHED_INDEX

        # ignore kpts with unknown depth data
        gt_matches0[~mask0] = self.IGNORE_INDEX
        gt_matches1[~mask1] = self.IGNORE_INDEX

        # also ignore point if its nearest neighbor is invalid
        gt_matches0[~mask1.gather(1, nn_matches0)] = self.IGNORE_INDEX
        gt_matches1[~mask0.gather(1, nn_matches1)] = self.IGNORE_INDEX

        return {
            **data,
            'keypoints0': kpts0,
            'keypoints1': kpts1,
            'descriptors0': desc0,
            'descriptors1': desc1,
            'scores0': scores0,
            'scores1': scores1,
            'gt_matches0': gt_matches0,
            'gt_matches1': gt_matches1
        }


if __name__ == '__main__':
    from datasets.megadepth import MegaDepthWarpingDataset, MegaDepthPairsDataset
    from models.superglue import SuperGlue

    device = torch.device('cpu')

    matches_generator = SuperPointMatchesGenerator(
        config=dict(
            max_keypoints=1024,
            keypoint_threshold=0,
            gt_positive_threshold=5,
            gt_negative_threshold=15
        )
    )
    matches_generator.eval().to(device)
    # superglue = SuperGlue(dict(
    #     weights='/home/ostap/projects/DepthGlue/models/weights/superglue_outdoor.pth'
    # ))
    # superglue.to(device)

    with open('../assets/megadepth_train_scenes.txt') as f:
        scenes_list = f.readlines()
    scenes_list = [s.rstrip() for s in scenes_list]

    ds = MegaDepthPairsDataset(
        root_path='/datasets/extra_space2/ostap/MegaDepth',
        scenes_list=['0012'],
        target_size=(640, 480)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    data = next(iter(dl))
    data = data_to_device(data, device)

    data = matches_generator(data)
    # loss = superglue.training_step(data)
    # print(loss)
