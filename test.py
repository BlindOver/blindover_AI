import os
import argparse
import time
from typing import *

import torch
import torch.nn as nn

from utils.dataset import load_dataloader
from utils.plots import plot_results
from quantization.quantization import converting_quantization


def test(
    test_loader,
    device,
    model: nn.Module,
    project_name: Optional[str] = None,
    measure_latency: bool=False,
):
    image_list, label_list, output_list = [], [], []
    
    model.eval()
    if measure_latency:
        start = time.time()

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

    if project_name is not None:
        plot_results(image_list, label_list, output_list, project_name)
    print(f'{"="*20} Test Results: Accuracy {acc*100:.2f} {"="*20}')

    if measure_latency:
        print(f'time: {time.time()-start:.3f}')


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
    parser.add_argument('--quantization', action='store_true',
                        help='evaluate the performance of quantized model')
    parser.add_argument('--measure_latency', action='store_true',
                        help='measure latency time')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                        help='set device for inference')
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
    device = torch.device(args.device)

    q = args.quantization

    # load model
    if args.model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=args.num_classes, pre_trained=False, quantize=q)
        
    elif args.model_name == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=args.num_classes, pre_trained=False)

    elif args.model_name == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=args.num_classes, pre_trained=False)

    elif args.model_name == 'resnet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=args.num_classes, quantize=q)

    elif args.model_name == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=args.num_classes, quantize=q)

    else:
        raise ValueError(f'model name {args.model_name} does not exists.')

    model.load_state_dict(torch.load(args.weight, map_location=device))

    if q:
        model = converting_quantization(model)

    model = model.to(device)

    test(
        test_loader,
        device=device,
        model=model,
        project_name=args.project_name,
        measure_latency=args.measure_latency,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)