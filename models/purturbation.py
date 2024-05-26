import torch
import torchvision
from torch import nn
from d2l import torch as d2l

class Perturbation:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def add_gaussian_noise(self, tensor):
        # Tensor dimension = (b,h,w,1)
        if tensor.ndim != 4 or tensor.size(-1) != 1:
            raise ValueError("Input tensor dimension error")
        
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        
        # It is binary mask tensor
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)

        return noisy_tensor
    
