import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

class ImageLoader:

    def __init__(self, 
        data_folder_path: str, 
        batch_size: int, 
        image_size: int, 
        device: torch.device
    ) -> None:
        """Creates Loader object, that loads images from given folder.

        Args:
            data_folder_path (str): folder that the images are in, images must be in subfolder
            batch_size (int): batch size to load the images in
            image_size (int): size to rescale the images to 
            device (torch.device | str, optional): device that the images should be returned on. can be str or torch.device. Defaults to "cpu".
        """
        assert isinstance(data_folder_path, str) and os.path.isdir(data_folder_path), "Path is invalid or does not exist"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size is invalid"
        assert isinstance(image_size, int) and image_size > 0, "Image size is invalid"
        assert isinstance(device, torch.device), "device must be of type torch.device"

        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_folder_path = data_folder_path

        self.loader = self.__create_loader(data_folder_path, batch_size, image_size)
        self.data_iterator = iter(self.loader)

    def __create_loader(self, data_folder_path: str, batch_size: int, image_size: int) -> DataLoader:
        transform_pipeline = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])
        
        dataset = torchvision.datasets.ImageFolder(
            root=data_folder_path, 
            transform=transform_pipeline
        )
        return DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    def get_batch(self) -> torch.Tensor:
        try:
            images, _ = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.loader) # reset iterator
            images, _ = next(self.data_iterator)
        return images.to(self.device)

