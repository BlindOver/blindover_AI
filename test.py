import os
import argparse
import time
from typing import *

import torch

from utils.dataset import load_dataloader
from utils.plots import plot_results
from quantization.quantization import load_model, quantization_serving
from quantization.utils import print_latency


def test(
    test_loader,
    device,
    model: nn.Module,
    project_name: str,
):
    image_list, label_list, output_list = [], [], []
    
    model.eval()
    with torch.no_grad():
        batch_acc = 0
        for batch, (images, labels) in enumerate(test_loader):
            image_list.append(images)
            label_list.append(labels)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            output_index = torch.argmax(outputs, dim=1)
            output_list.append(output_index.cpu())
            acc = (output_index == labels).sum() / (len(outputs))

            batch_acc += acc.item()
    
    plot_results(image_list, label_list, output_list, project_name)
    print(f'{"="*20} Test Results: Accuracy {acc*100:.2f} {"="*20}')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Model', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--subset', type=str, default='valid',
                        help='dataset subset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name consisting of mobilenet, shufflenet, efficientnet, resnet18 and resnet50')
    parser.add_argument('--weight', type=str, required=True,
                        help='load trained model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size before applying cropping')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers in cpu')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch Size for training model')
    parser.add_argument('--num_classes', type=int, default=33,
                        help='class number of dataset')
    parser.add_argument('--project_name', type=str, default='prj',
                        help='create new folder named project name')
    parser.add_argument('--quantization', store='action_treu',
                        help='evaluate the performance of quantized model')
    parser.add_argument('--measure_latency', store='action_true',
                        help='measure latency time')
    return parser


def main(args):
    
    os.makedirs(f'./runs/test/{args.project_name}', exist_ok=True)
    
    test_loader = load_dataloader(
        path=args.data_path,
        img_size=args.img_size,
        subset=args.subset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    # setting device
    if args.quantization:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    if args.quantization:
        model = quantization_serving(model_name=args.model_name, weight=args.weight, num_classes=args.num_classes)
    else:
        model = load_model(model_name=args.model_name, num_classes=args.num_classes, quantization=False)
        model.load_state_dict(torch.load(args.weight))

    model = model.to(device)

    if args.measure_latency:
        print_latency(
            test(test_loader, device=device, model=model, project_name=args.project_name),
            req_return=False,
        )

    else:
        test(test_loader, device=device, model=model, project_name=args.project_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)