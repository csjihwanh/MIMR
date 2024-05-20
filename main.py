from models import MIMR
from dataloader import create_dataloader
from utils import train

import argparse
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('directory', type=str, help='Path to the directory containing image files')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the data')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mode', type=int, default=0, help='0: train, 1: evaluation')
    parser.add_argument('--weight_path', type=str, default=None, help='path to weight')

   
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    assert args.batch_size == 1, 'batch size must be 1.'

    dataloader = create_dataloader(
        directory=args.directory,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    model = MIMR(1,1)
    model.cuda()
    
    if args.mode == 0:
        train(model,dataloader,epochs=10,lr=1e-4,graph_loss_weight=0.1,device='cuda')
