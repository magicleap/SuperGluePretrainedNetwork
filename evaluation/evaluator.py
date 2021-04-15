import numpy as np
from pathlib import Path
import torch

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, AverageTimer, pose_auc,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from models.matching import Matching


class Evaluator:
    def __init__(self, models_config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = models_config
        self.pose_errors = []
        self.precisions = []
        self.matching_scores = []

    def evaluate(self, dataset):
        print('Running inference on device \"{}\"'.format(self.device))
        timer = AverageTimer(newline=True)

        matching = Matching(self.config).eval().to(self.device)

        for i, data_dict in enumerate(dataset):
            # Load the image pair
            img1, img2 = data_dict['image0'].unsqueeze(0).to(self.device), data_dict['image1'].unsqueeze(0).to(self.device)
            transf_type, H = data_dict['transformation']['type'], data_dict['transformation']['H']

            # Perform the matching.
            pred = matching({'image0': img1, 'image1': img2})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            # Estimate the pose and compute the pose error.
            K0 = data_dict['transformation']['K0']
            K1 = data_dict['transformation']['K1']
            T_0to1 = data_dict['transformation']['T_0to1'].astype(float).reshape(4, 4)

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
            self.pose_errors.append(pose_error)
            self.precisions.append(precision)
            self.matching_scores.append(matching_score)

            timer.update('eval')

        res_dict = self.calculate_metrics()

        return res_dict

    def calculate_metrics(self, thresholds=None):
        if thresholds is None:
            thresholds = [5, 10, 20]

        aucs = pose_auc(self.pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        prec = 100. * np.mean(self.precisions)
        ms = 100. * np.mean(self.matching_scores)

        return {'AUC@5': aucs[0], 'AUC@10': aucs[1], 'AUC@20': aucs[2], 'Prec': prec, 'MScore': ms}
