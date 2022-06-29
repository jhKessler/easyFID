import os
import torch
import numpy as np
from skimage.io import imsave
from easyFID.image_loader import ImageLoader
from tempfile import TemporaryDirectory

# save random images for testing
tmp_dir = TemporaryDirectory()
os.makedirs(f"{tmp_dir.name}/images")
for i in range(100):
    imsave(os.path.join(tmp_dir.name, "images", f"image_{i}.png"), np.random.rand(100, 100, 3))

def test_loader_creation():
    loader = ImageLoader(tmp_dir.name, batch_size=16, image_size=100, device=torch.device("cpu"))
    assert isinstance(loader, ImageLoader)
    assert isinstance(loader.loader, torch.utils.data.DataLoader)
    assert isinstance(loader.device, torch.device)
    
def test_loader_get_batch_cpu():
    loader = ImageLoader(tmp_dir.name, batch_size=16, image_size=100, device=torch.device("cpu"))
    images = loader.get_batch()
    assert isinstance(images, torch.Tensor)
    assert images.shape == (16, 3, 100, 100)
    assert images.device.type == "cpu"

def test_loader_get_batch_cpu():
    if not torch.cuda.is_available():
        return
    loader = ImageLoader(tmp_dir.name, batch_size=16, image_size=100, device=torch.device("cuda"))
    images = loader.get_batch()
    assert isinstance(images, torch.Tensor)
    assert images.shape == (16, 3, 100, 100)
    assert images.device.type == "cuda"