
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
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels')


   
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args:
        directory = args.directory
        batch_size = args.batch_size
        shuffle = args.shuffle
        num_workers = args.num_workers
        epochs = args.epochs
        lr = args.lr
        mode = args.mode
        weight_path = args.weight_path
        num_classes = args.num_classes
        num_channels = args.num_channels
    else: 
        print('No argument detected. EXIT')
        exit()

    assert batch_size == 1, 'batch size must be 1.'

    dataloader = create_dataloader(
        directory=directory,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    model = MIMR(n_channels = num_channels, n_classes=num_classes)
    model = model.cuda()
    
    if args.mode == 0:
        train(model,dataloader,epochs=10,lr=1e-4,graph_loss_weight=0.1,device='cuda')
