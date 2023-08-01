import sys
sys.path.append('/home/hoo7311/anaconda3/envs/yolov7/lib/python3.8/site-packages')

import os
import argparse
import time
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn

from utils.dataset import load_dataloader
from utils.plots import plot_results
from quantization.quantize import (
    converting_quantization, 
    ptq_serving, 
    qat_serving, 
    fuse_modules, 
    print_size_of_model,
)


classes = {
    0: '2%', 1: '박카스', 2: '칠성 사이다', 3: '칠성 사이다 제로', 4: '초코 우유',
    5: '코카 콜라', 6: '데미소다 사과', 7: '데미소다 복숭아', 8: '솔의눈', 9: '환타 오렌지',
    10: '게토레이', 11: '제티', 12: '맥콜', 13: '우유', 14: '밀키스', 15: '밀키스 제로',
    16: '마운틴 듀', 17: '펩시', 18: '펩시 제로', 19: '포카리 스웨트', 20: '파워에이드',
    21: '레드불', 22: '식혜', 23: '스프라이트', 24: '스프라이트 제로', 25: '딸기 우유',
    26: '비타 500', 27: '브이톡 블루레몬', 28: '브이톡 복숭아', 29: '웰치스 포도',
    30: '웰치스 오렌지', 31: '웰치스 화이트그레이프',32: '제로 콜라',
}


count_classes = {k: [0, 0] for k, v in classes.items()}


def test(
    test_loader,
    device,
    model: nn.Module,
    project_name: Optional[str]=None,
    plot_result: bool=False,
):
    if (project_name is None) and (not plot_result):
        raise ValueError('define project name')

    if plot_result:
        image_list, label_list, output_list = [], [], []
    
    start = time.time()

    model.eval()
    model = model.to(device)
    batch_acc = 0
    with torch.no_grad():
        for batch, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if plot_result:
                image_list.append(images)
                label_list.append(labels)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            output_index = torch.argmax(outputs, dim=1)

            # calculate the accuracy for each class
            for idx, output in enumerate(output_index):
                count_classes[labels[idx].item()][1] += 1 # count label classes
                if labels[idx] == output:
                    count_classes[output.item()][0] += 1 # count predicted classes
                
            if plot_result:
                output_list.append(output_index.cpu())

            acc = (output_index == labels).sum() / len(outputs)
            batch_acc += acc.item()
    
    print(f'{"="*20} Inference Time: {time.time()-start:.3f}s {"="*20}')    
    
    if project_name is not None:
        if plot_result:
            plot_results(image_list, label_list, output_list, project_name)
    
    print(f'{"="*20} Test Average Accuracy {batch_acc/(batch+1)*100:.2f} {"="*20}')
    for k, v in count_classes.items():
        print('{0: ^15s} --> accuracy: {1:.3f}%, {2}/{3}'.format(
            classes[k], (v[1]+1e-7)/v[0], v[1], v[0]))


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
    parser.add_argument('--quantization', type=str, default='none', choices=['none', 'qat', 'ptq'],
                        help='evaluate the performance of quantized model or float32 model when setting none')
    parser.add_argument('--plot_result', action='store_true',
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

    q = True if args.quantization != 'none' else False

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
        model = resnet18(num_classes=args.num_classes, pre_trained=False, quantize=q)

    elif args.model_name == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=args.num_classes, pre_trained=False, quantize=q)

    else:
        raise ValueError(f'model name {args.model_name} does not exists.')

    # quantization
    if args.quantization == 'ptq':
        model = ptq_serving(model=model, weight=args.weight)

    elif args.quantization == 'qat':
        model = qat_serving(model=model, weight=args.weight)

    else: # 'none'
        model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    test(
        test_loader,
        device=device,
        model=model,
        project_name=args.project_name,
        plot_result=args.plot_result,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)