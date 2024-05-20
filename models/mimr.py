import torch
import torch.nn as nn 

from .unet import UNet
from .graph import MIMRGNN

class MIMR(nn.Module):
    def __init__(
        self, 
        n_channels :int, 
        n_classes:int
    ):
        super().__init__()
        self.unet = UNet(n_classes=n_classes, n_channels=n_channels)
        self.gnn = MIMRGNN(layer_num=10,inter_dim=64,out_dim=64)

    def forward(self, x):
        recon_x = self.unet(x)
        binary_recon_x = torch.round(recon_x)
        graph_feature = self.gnn(binary_recon_x)

        return recon_x, graph_feature


if __name__ == "__main__":
    binary_map = torch.randint(0, 2, (224, 224), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=1)

    test = MIMR(1,1)
    test(binary_map)