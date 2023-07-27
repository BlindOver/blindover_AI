import os

import torch

from quantization import prepare_ptq, converting_quantization


# only apply resnet based model
def fuse_modules(model):
    model = model.cpu()
    model.eval()
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

    return torch.quantization.fuse_modules(model, modules)


def print_size_of_model(model, label=''):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    print('model: ', label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def comparison_size_of_models(model_name: str, num_classes: int=33):
    if model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        float_model = ShuffleNetV2(num_classes=33, pre_trained=False, quantize=True)
        
    elif model_name == 'resnet18':
        from models.resnet import resnet18
        float_model = resnet18(num_classes=33, quantize=True)
        
    elif model_name == 'resnet50':
        from models.resnet import resnet50
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