import torch
import argparse
parser = argparse.ArgumentParser(description='PyTorch load checkpoint')
parser.add_argument('--dir', metavar='DIR', default='checkpoint.pth.tar',
                    help='path to checkpoint (default: checkpoint.pth.tar)')
args = parser.parse_args()
dir = args.dir
checkpoint = torch.load(dir, map_location=torch.device('cpu'))
epoch = checkpoint['epoch']
arch = checkpoint['arch']
best_acc1 = checkpoint['best_acc1']
best_acc5 = checkpoint['best_acc5']

print(f'epoch_finished: {epoch}, the arch is {arch} and the best acc1 & acc5 are {best_acc1}, {best_acc5}')
