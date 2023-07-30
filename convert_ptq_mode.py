import os
import argparse

import torch
import torch.nn as nn

from utils.dataset import load_dataloader
from models.shufflenet import ShuffleNetV2
from models.resnet import resnet18, resnet50
from quantization.quantize import (
    prepare_ptq, 
    fuse_modules, 
    converting_quantization, 
    calibration_for_quantization,
)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Converting on PTQ mode', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='dataset path')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name consisting of shufflenet, resnet18 and resnet50')
    parser.add_argument('--weight', type=str, required=True,
                        help='load trained model')
    parser.add_argument('--num_classes', type=int, default=33,
                        help='the number of classes')
    parser.add_argument('--backend', type=str, default='x86',
                        help='the number of classes')
    
    return parser


def main(args):
    name = args.model_name

    if name == 'shufflenet':
        model = ShuffleNetV2(num_classes=args.num_classes, pre_trained=False, quantize=True)

    elif name == 'resnet18':
        model = resnet18(num_classes=args.num_classes, pre_trained=False, quantize=True)

    elif name == 'resnet50':
        model = resnet50(num_classes=args.num_classes, pre_trained=False, quantize=True)
    
    else:
        raise ValueError(f'{name} does not exists')

    data_loader = load_dataloader(
        path=args.data_path,
        subset='train',
        batch_size=1,
    )

    print('Start Quantizatoin...!')
    model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
    model = fuse_modules(model, mode='eval')
    model = prepare_ptq(model, backend=args.backend)
    model = calibration_for_quantization(model, data_loader=data_loader)
    model = converting_quantization(model)
    print('Complete Quantization...!')

    origin_weight_path = args.weight
    origin_file_name = origin_weight_path.split('/')[-1].split('.')[0]
    quantized_file_name = 'quantized_' + origin_file_name
    return_file_name = origin_weight_path.replace(origin_file_name, quantized_file_name)

    torch.save(model.state_dict(), return_file_name)
    print('Complete Saving quantized weight...!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Converting (PTQ)', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)