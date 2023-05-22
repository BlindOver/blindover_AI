import os
import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.dataset import load_dataloader
from utils.callback import CheckPoint, EarlyStopping


def test(
    test_loader,
    device: str,
    model: nn.Module,
):
    model.eval()
    with torch.no_grad():
        batch_acc = 0
        for batch, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index == labels).sum() / (len(outputs))

            batch_acc += acc.item()
    
    print(f'{"="*20} Test Results: Accuracy {acc*100:.2f} {"="*20}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Model', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--subset', type=str, default='valid',
                        help='dataset subset')
    parser.add_argument('--model', type=str, required=True,
                        help='model name consisting of mobilenet, shufflenet, mnasnet and efficientnet')
    parser.add_argument('--weight', type=str, required=True,
                        help='load trained model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size before applying cropping')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers in cpu')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch Size for training model')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='class number of dataset')
    
    return parser


def main(args):
    
    test_loader = load_dataloader(
        path=args.data_path,
        img_size=args.img_size,
        subset=args.subset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=args.num_classes, pre_trained=False)

    elif args.model == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=args.num_classes, pre_trained=False)

    elif args.model == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=args.num_classes, pre_trained=False)

    elif args.model == 'mnasnet':
        from models.mnasnet import MNASNet
        model = MNASNet(num_classes=args.num_classes, pre_trained=False)

    else:
        raise ValueError(f'{args.model} does not exists')

    model.load_state_dict(torch.load(args.weight))
    model = model.to(device)

    test(test_loader, device=device, model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
