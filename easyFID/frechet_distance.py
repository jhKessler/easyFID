from tkinter import Image
from typing import Callable, Union
import torch
import torch.nn as nn
from torchvision import transforms

from .image_loader import ImageLoader
from .fid_calculator import FIDCalculator
from.inception_model import InceptionModel

class FrechetDistance:
    
    @staticmethod
    def _inspect_gen_fn(generation_fn: Callable, device: torch.device) -> tuple[int, int]:
        """Inspects the generation function to get the batch size and image size
            throws Error if a requirement of the function is violated

        Args:
            generation_fn (Callable): Function that generates synthetic images when called without arguments, make sure output is in range 0-255
        
        Returns:
            tuple[int, int]: batch size and image size
        """
        images = generation_fn()
        assert len(images.shape) == 4, "Generation function must return a batch of images"
        assert images.device == device, "Generation function must return images on the same device as the device argument"
        batch_size, image_size, _, _ = images.shape
        return batch_size, image_size

    def calculate(self,
        generation_fn: Callable,
        real_images_folder: str,
        *,
        device: Union[str, torch.device] = "cpu",   
        n_images: int = 10_000) -> float:
        """Calculates Frechet Inception Distance between two distributions

        Args:
            generation_fn (Callable): Function that generates synthetic images when called without arguments, make sure output is in range 0-255 and on chosen device
            real_images_folder (str): Superfolder that the real images are in
            device (Union[str, torch.device]): device to run the calculations on, can be str or torch.device. Defaults to "cpu".
            n_images (int, optional): number of images to calc fid score on. Defaults to 10_000.
        
        Returns:
            float: Frechet Inception Distance
        """
        if isinstance(device, str):
            device = torch.device(device)
        batch_size, image_size = self._inspect_gen_fn(generation_fn, device)
        inception_model = InceptionModel().to(device)
        real_loader = ImageLoader(real_images_folder, batch_size, image_size, device)

        real_inceptions = self._get_n_inception_scores(real_loader.get_batch, inception_model.forward, n_images)
        fake_inceptions = self._get_n_inception_scores(generation_fn, inception_model.forward, n_images)

        return real_inceptions, fake_inceptions

    def _get_n_inception_scores(self, img_fn: Callable, inception_fn: Callable, n: int) -> torch.Tensor:
        assert isinstance(n, int) and n > 0
        imgs = inception_fn(img_fn())
        while len(imgs) < n:
            if not len(next_batch := img_fn()):
                raise RuntimeError(f"img function isn't returning anything!")
            imgs = torch.cat([imgs, inception_fn(next_batch)]) # add batch until we reach n images
        return imgs[:n]