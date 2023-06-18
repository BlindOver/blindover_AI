import argparse
from PIL import Image
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Padding
from models.efficientnet import EfficientNetV2
from models.mnasnet import MNASNet
from models.mobilenet import MobileNetV3
from models.shufflenet import ShuffleNetV2


classes = {
    0: '2%', 1: '박카스', 2: '칠성 사이다', 3: '칠성 사이다 제로', 4: '초코 우유',
    5: '코카 콜라', 6: '데미소다 사과', 7: '데미소다 복숭아', 8: '솔의눈', 9: '환타 오렌지',
    10: '게토레이', 11: '제티', 12: '맥콜', 13: '우유', 14: '밀키스', 15: '밀키스 제로',
    16: '마운틴 듀', 17: '펩시', 18: '펩시 제로', 19: '포카리 스웨트', 20: '파워에이드',
    21: '레드불', 22: '식혜', 23: '스프라이트', 24: '스프라이트 제로', 25: '딸기 우유',
    26: '비타 500', 27: '브이톡 블루레몬', 28: '브이톡 복숭아', 29: '웰치스 포도',
    30: '웰치스 오렌지', 31: '웰치스 화이트그레이프',32: '제로 콜라',
}


def load_image(src: str):    
    img = Image.open(src).convert('RGB')
    img = transformation(img)
    img = img.unsqueeze(dim=0)
    return img, src


def inference(src: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(src)
        prob = F.softmax(outputs)
        result = classes[torch.argmax(prob, dim=1)]
    return result


def load_model(model_name: str, weight: Optional[str]):
    if model_name == 'mobilenet':
        model = MobileNetV3(num_classees=33, pre_trained=False)

    elif model_name == 'shufflenet':
        model = ShuffleNetV2(num_classes=33, pre_trained=False)

    elif model_name == 'mnasnet':
        model = MNASNet(num_classes=33, pre_trained=False)
    
    elif model_name == 'efficientnet':
        model = EfficientNetV2(num_classes=33, pre_trained=False)

    else:
        raise ValueError(f'{model_name} does not exists')

    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    
    return model


def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference', add_help=False)
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('--src', type=str, required=True,
                        help='input image')
    parser.add_argument('--weight', type=str, required=True,
                        help='a path of trained weight file')
    parser.add_argument('--fill_color', type=int, default=0,
                        help='RGB color')
    parser.add_argument('--num_classes', type=int, default=33,
                        help='the number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size')
    return parser


def main(args):
    transformation = transforms.Compose([
        Padding(fill=(args.fill_color, args.fill_color, args.fill_color)),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    model = load_model(args.model_name, args.weight)
    img, src = load_image(args.src)
    result = inference(src, model)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)