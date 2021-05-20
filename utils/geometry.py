# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import numpy as np
import cv2


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    # assert E is not None
    if E is None:
        return None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w - 1 - cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w - 1 - cx],
                         [0., fy, h - 1 - cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h - 1 - cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1. / scales[0], 1. / scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
                      + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]

        aucs.append(np.trapz(r, x=e) / t)
    return aucs
