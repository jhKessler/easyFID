from typing import Dict, Iterable, Callable
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
class InceptionModel(nn.Module):

    @staticmethod
    def _get_imagenet_model() -> Callable:
        """Creates image pipeline for inception network

        Args:
            device (torch.device): device to run on

        Returns:
            Callable: function that takes a batch of images and returns the inception features
        """
        image_net = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        image_net.eval()
        return image_net
        
    def __init__(self):
        super().__init__()
        self.model = self._get_imagenet_model()
        self.inception_output = None
        self.transforms_pipeline = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        layer = dict([*self.model.named_modules()])["avgpool"] # get inception layer
        layer.register_forward_hook(self.save_output_hook())

    def save_output_hook(self) -> Callable:
        def fn(_, __, output):
            self.inception_output = output
        return fn

    def forward(self, inp_images: torch.Tensor) -> torch.Tensor:
        _ = self.model(self.transforms_pipeline(inp_images))
        return self.inception_output.squeeze().squeeze()