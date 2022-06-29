import torch

from easyFID.frechet_distance import FrechetDistance

frechet = FrechetDistance()

def test_inception_fn():
    inception_fn = frechet._calc_inception_features_fn(torch.device("cpu"))
    images = inception_fn(torch.rand(2, 3, 64, 64))
    assert images.shape == (2, 2048)