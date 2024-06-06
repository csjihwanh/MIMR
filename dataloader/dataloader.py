import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms

class MaskDataset(Dataset):
    def __init__(self, directory, extensions=['.png', '.jpg', '.jpeg']):
        self.directory = directory
        self.extensions = extensions
        self.image_paths = self._get_all_image_files(directory)
        self.transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        print('Dataset size: ', self.__len__())

    def _get_all_image_files(self, directory):
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.extensions):
                    image_files.append(os.path.join(root, file))
        return image_files[:5]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') 
        image = (np.array(image) > 127).astype(np.int8) # check if pixel value is larger than 127 (half of 255)
        image = self.transform(image).to(torch.float32)
        return image

def create_dataloader(directory, batch_size=32, shuffle=True, num_workers=2):
    dataset = MaskDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
