import torch
import numpy as np
import pytorch_lightning as pl

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from utils.train_utils import min_stack, arange_like, get_inverse_transformation, reproject_keypoints
from utils.losses import criterion
from utils.geometry import estimate_pose, compute_pose_error, compute_epipolar_error, pose_auc
from utils.augmentation import SuperAugmentation


class MatchingModule(pl.LightningModule):
    UNMATCHED_INDEX = -1  # index of keypoint that don't have a match
    IGNORE_INDEX = -2  # index of keypoints to ignore during loss calculation

    def __init__(self, train_config, superpoint_config, superglue_config):
        super(MatchingModule, self).__init__()
        self.config = train_config
        self.superpoint = SuperPoint(superpoint_config)
        self.superglue = SuperGlue(superglue_config)
        self.aug = SuperAugmentation()

    def generate_gt_matches(self, data):
        """Given image pair, keypoints detected in each image, return set of ground truth correspondences"""
        gt_positive_threshold = self.config['gt_positive_threshold']
        gt_negative_threshold = self.config.get('gt_negative_threshold', gt_positive_threshold)

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

        gt_matches0[cross_check_consistent0][symmetric_dist > gt_positive_threshold] = self.IGNORE_INDEX
        gt_matches0[cross_check_consistent0][symmetric_dist > gt_negative_threshold] = self.UNMATCHED_INDEX

        gt_matches1[cross_check_consistent1][symmetric_dist > gt_positive_threshold] = self.IGNORE_INDEX
        gt_matches1[cross_check_consistent1][symmetric_dist > gt_negative_threshold] = self.UNMATCHED_INDEX

        # ignore kpts with unknown depth data
        gt_matches0[~mask0] = self.IGNORE_INDEX
        gt_matches1[~mask1] = self.IGNORE_INDEX

        # also ignore point if its nearest neighbor is invalid
        gt_matches0[~mask1.gather(1, nn_matches0)] = self.IGNORE_INDEX
        gt_matches1[~mask0.gather(1, nn_matches1)] = self.IGNORE_INDEX

        data = {
            **data,
            'keypoints0': kpts0, 'keypoints1': kpts1,
            'descriptors0': desc0, 'descriptors1': desc1,
            'scores0': scores0, 'scores1': scores1,
        }

        y_true = {
            'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1
        }

        return data, y_true

    def augment(self, batch):
        image0, transform0 = self.aug(batch['image0'])
        image1, transform1 = self.aug(batch['image1'])

        batch['image0'] = image0
        batch['image1'] = image1

        if 'K0' in batch['transformation']:
            batch['transformation']['K0'] = torch.matmul(transform0, batch['transformation']['K0'])
        if 'K1' in batch['transformation']:
            batch['transformation']['K1'] = torch.matmul(transform1, batch['transformation']['K1'])
        return batch

    def training_step(self, batch, batch_idx):
        self.superpoint.eval()

        # batch = self.augment(batch)

        with torch.no_grad():
            data, y_true = self.generate_gt_matches(batch)

        y_pred = self.superglue(data)

        loss = criterion(y_true, y_pred, margin=self.config['margin'])

        self.log('Train NLL loss', loss['loss'].detach(), prog_bar=True, sync_dist=True)
        self.log('Train Metric loss', loss['metric_loss'].detach(), prog_bar=True, sync_dist=True)

        return self.config['nll_weight'] * loss['loss'] + self.config['metric_weight'] * loss['metric_loss']

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Load intrinsics, translation vector and rotation matrix
        K0 = batch['transformation']['K0'][0].cpu().numpy()
        K1 = batch['transformation']['K1'][0].cpu().numpy()

        R = batch['transformation']['R'][0].cpu().numpy()
        T = batch['transformation']['T'][0].cpu().numpy()
        R = np.vstack([R, [0, 0, 0]])
        T_0to1 = np.column_stack([R, np.append(T, [[1]])])

        # Estimate the pose and compute the pose error.
        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        pose_error = np.maximum(err_t, err_R)
        self.val_pose_error.append(pose_error)
        self.log('Val Precision', 100. * precision, sync_dist=True, on_epoch=True)
        self.log('Val Matching Score', 100. * matching_score, sync_dist=True, on_epoch=True)

    def on_validation_epoch_start(self):
        self.val_pose_error = []

    def on_validation_epoch_end(self):
        thresholds = [5, 10, 20]

        aucs = pose_auc(self.val_pose_error, thresholds)
        aucs = [100. * yy for yy in aucs]

        self.log_dict({'Val AUC@5': aucs[0], 'Val AUC@10': aucs[1], 'Val AUC@20': aucs[2]}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.superglue.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1,
            gamma=self.config['scheduler_gamma']
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def forward(self, data):
        # run superpoint on both images
        pred = {}
        pred0 = self.superpoint({'image': data['image0']})
        pred = {**pred, **{k + '0': v for k, v in pred0.items()}}

        pred1 = self.superpoint({'image': data['image1']})
        pred = {**pred, **{k + '1': v for k, v in pred1.items()}}

        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # predict matching scores
        scores = self.superglue(data)['scores']

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return {
            **pred,
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
