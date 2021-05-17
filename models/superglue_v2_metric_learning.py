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
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import numpy as np

from training.train_utils import pairwise_cosine_dist


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        # load weights is path is given
        if self.config.get('weights', None) is not None:
            path = self.config['weights']
            self.load_state_dict(torch.load(str(path), map_location='cpu'))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def _get_matching_scores(self, data):

        # Keypoint normalization.
        kpts0 = normalize_keypoints(data['keypoints0'], data['image0'].shape)
        kpts1 = normalize_keypoints(data['keypoints1'], data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = data['descriptors0'] + self.kenc(kpts0, data['scores0'])
        desc1 = data['descriptors1'] + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores= log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        return {
            'scores': scores,
            'mdesc0': mdesc0,
            'mdesc1': mdesc1
        }

    def forward(self, data):
        """Run SuperGlue inference on a pair of keypoints and descriptors"""
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        pred = self._get_matching_scores(data)
        scores = pred['scores']

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
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            **pred
        }

    def training_step(self, data):
        """Training step of superglue model. Returns loss."""

        if data['keypoints0'].size(1) == 0 or data['keypoints1'].size(1) == 0:
            return {
                'skip_train_step': True
            }

        pred = self._get_matching_scores(data)
        scores = pred['scores']
        mdesc0, mdesc1 = pred['mdesc0'], pred['mdesc1']

        dist = pairwise_cosine_dist(mdesc0.permute(0, 2, 1), mdesc1.permute(0, 2, 1))
        gt_matches0, gt_matches1 = data['gt_matches0'], data['gt_matches1']

        # loss for keypoints with gt match
        batch_idx, idx_kpts0 = torch.where(gt_matches0 >= 0)
        _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
        mean_weights = (1 / counts)[inv_idx]

        idx_kpts1 = gt_matches0[batch_idx, idx_kpts0]
        matched_loss = (-scores[batch_idx, idx_kpts0, idx_kpts1] * mean_weights).sum()
        matched_triplet_loss = self.get_matched_triplet_loss(
            dist=dist, indexes=(batch_idx, idx_kpts0, idx_kpts1),
            mean_weights=mean_weights
        )

        # loss for unmatched keypoints in the image 0
        batch_idx, idx_kpts0 = torch.where(gt_matches0 == -1)
        _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
        mean_weights = (1 / counts)[inv_idx]

        unmatched0_loss = (-scores[batch_idx, idx_kpts0, -1] * mean_weights).sum()
        unmatched0_margin_loss = self.get_unmatched_margin_loss(
            dist=dist, indexes=(batch_idx, idx_kpts0),
            mean_weights=mean_weights, zero_to_one=True
        )

        # loss for unmatched keypoints in the image 1
        batch_idx, idx_kpts1 = torch.where(gt_matches1 == -1)
        _, inv_idx, counts = torch.unique_consecutive(batch_idx, return_inverse=True, return_counts=True)
        mean_weights = (1 / counts)[inv_idx]

        unmatched1_loss = (-scores[batch_idx, -1, idx_kpts1] * mean_weights).sum()
        unmatched1_margin_loss = self.get_unmatched_margin_loss(
            dist=dist, indexes=(batch_idx, idx_kpts1),
            mean_weights=mean_weights, zero_to_one=False
        )
        return {
            # 'sinkhorn_scores': scores,
            # 'mdesc0': mdesc0,
            # 'mdesc1': mdesc1,
            'skip_train_step': False,
            'loss': (matched_loss + 0.5 * (unmatched0_loss + unmatched1_loss)) / scores.size(0),
            'triplet_loss': (matched_triplet_loss + unmatched0_margin_loss + unmatched1_margin_loss) / scores.size(0)
        }

    def get_matched_triplet_loss(self, dist, indexes, mean_weights):
        margin = self.config['triplet_margin']
        if margin == -1:
            return torch.tensor(0, device=dist.device)
        batch_idx, idx_kpts0, idx_kpts1 = indexes
        # distance between anchor and positive
        dist_ap = dist[batch_idx, idx_kpts0, idx_kpts1]

        dist_detached = dist.detach().clone()
        dist_detached[batch_idx, idx_kpts0, idx_kpts1] = np.inf
        idx_kpts0_closest_to_1 = torch.argmin(dist_detached, dim=1)
        idx_kpts1_closest_to_0 = torch.argmin(dist_detached, dim=2)
        idx_kpts1_neg = idx_kpts1_closest_to_0[batch_idx, idx_kpts0]
        idx_kpts0_neg = idx_kpts0_closest_to_1[batch_idx, idx_kpts1]

        dist_an0 = dist[batch_idx, idx_kpts0, idx_kpts1_neg]
        dist_an1 = dist[batch_idx, idx_kpts1, idx_kpts0_neg]

        loss0 = torch.maximum(dist_ap - dist_an0 + margin, torch.tensor(0, device=dist.device))
        loss1 = torch.maximum(dist_ap - dist_an1 + margin, torch.tensor(0, device=dist.device))
        return (loss0 * mean_weights).sum() + (loss1 * mean_weights).sum()

    def get_unmatched_margin_loss(self, dist, indexes, mean_weights, zero_to_one=True):
        margin = self.config['triplet_margin']
        if margin == -1:
            return torch.tensor(0, device=dist.device)
        batch_idx, idx_kpts = indexes

        idx_kpts_closest = torch.argmin(dist, dim=2 if zero_to_one else 1)
        idx_kpts_neg = idx_kpts_closest[batch_idx, idx_kpts]

        # distance anchor-negative
        if zero_to_one:
            dist_an = dist[batch_idx, idx_kpts, idx_kpts_neg]
        else:
            dist_an = dist[batch_idx, idx_kpts_neg, idx_kpts]

        loss = torch.maximum(-dist_an + margin, torch.tensor(0, device=dist.device))
        return (loss * mean_weights).sum()



