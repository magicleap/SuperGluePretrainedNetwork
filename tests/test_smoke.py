import os

import torch

from superglue import Matching
from superglue.utils import read_image


def test_smoke():
    model = Matching(config={}).train(False)
    img_path = os.path.join(os.path.dirname(__file__), 'lena_color.png')
    img1 = read_image(img_path)
    assert img1.numpy().shape == (1, 1, 512, 512)
    img2 = torch.flip(img1, dims=(2,)) * .95
    with torch.no_grad():
        res = model({'image0': img1, 'image1': img2})
    assert set(res.keys()) == {'keypoints0', 'scores0', 'descriptors0', 'keypoints1', 'scores1', 'descriptors1',
                               'matches0', 'matches1', 'matching_scores0', 'matching_scores1'}
