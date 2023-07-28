import os
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn


# for post training quantization
def prepare_ptq(model: nn.Module, backend: str='x86'):
    model.eval()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    return torch.quantization.prepare(model)


# for quantization aware training
def prepare_qat(model: nn.Module, backend: str='x86'):
    model.train()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    return torch.quantization.prepare_qat(model)


# after training
def converting_quantization(model: nn.Module):
    model.eval()
    model = model.cpu()
    return torch.quantization.convert(model)


# for serving of ptq model
def ptq_serving(
    model: nn.Module,
    weight: str, # the path of weight file
    backend: str='x86',
):

    model = fuse_modules(model, mode='eval')
    model = prepare_ptq(model, backend)
    model = converting_quantization(model)
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    return model


# for serving of qat model
def qat_serving(
    model: nn.Module, 
    weight: str, 
    backend: str='x86',
):

    model = fuse_modules(model, mode='train')
    model = prepare_qat(model, backend)
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    model = converting_quantization(model)
    return model


def calibration_for_quantization(
    model,
    data_loader,
    device=torch.device('cpu'),
):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for image, _ in tqdm(data_loader, total=len(data_loader)):
            image = image.to(device)
            _ = model(image)
    
    return model


# only apply resnet based model
def fuse_modules(model: nn.Module, mode: str='eval'):
    assert mode in ('eval', 'train')
    model = model.cpu()
    model.eval() if mode == 'eval' else model.train()

    modules = [
        ['conv1', 'bn1'],
        ['layer1.0.conv1', 'layer1.0.bn1'],
        ['layer1.0.conv2', 'layer1.0.bn2'],
        ['layer1.1.conv1', 'layer1.1.bn1'],
        ['layer1.1.conv2', 'layer1.1.bn2'],
        ['layer2.0.conv1', 'layer2.0.bn1'],
        ['layer2.0.conv2', 'layer2.0.bn2'],
        ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
        ['layer2.1.conv1', 'layer2.1.bn1'],
        ['layer2.1.conv2', 'layer2.1.bn2'],
        ['layer3.0.conv1', 'layer3.0.bn1'],
        ['layer3.0.conv2', 'layer3.0.bn2'],
        ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
        ['layer3.1.conv1', 'layer3.1.bn1'],
        ['layer3.1.conv2', 'layer3.1.bn2'],
        ['layer4.0.conv1', 'layer4.0.bn1'],
        ['layer4.0.conv2', 'layer4.0.bn2'],
        ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
        ['layer4.1.conv1', 'layer4.1.bn1'],
        ['layer4.1.conv2', 'layer4.1.bn2'],
    ]

    try: # resnet based-model
        model = torch.quantization.fuse_modules(model, modules)
    
    except: # shufflenet
        pass

    return model


def print_size_of_model(model, label=''):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    print('model: ', label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def comparison_size_of_models(model_name: str, num_classes: int=33):
    if model_name == 'shufflenet':
        from ..models.shufflenet import ShuffleNetV2
        float_model = ShuffleNetV2(num_classes=33, pre_trained=False, quantize=True)
        
    elif model_name == 'resnet18':
        from ..models.resnet import resnet18
        float_model = resnet18(num_classes=33, quantize=True)
        
    elif model_name == 'resnet50':
        from ..models.resnet import resnet50
        float_model = resnet50(num_classes=33, quantize=True)

    else:
        raise ValueError(f'model name {model_name} does not exists.')
    
    prepared_model = prepare_ptq(float_model)
    quantized_model = converting_quantization(prepared_model)
    
    float_model.eval()
    float_model = float_model.cpu()
    quantized_model.eval()
    quantized_model = quantized_model.cpu()
    
    f = print_size_of_model(float_model, 'float32')
    q = print_size_of_model(quantized_model, 'int8')
    print("{0:.2f} times smaller".format(f / q))