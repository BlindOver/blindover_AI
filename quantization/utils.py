import os

import torch
import torchvision.models as models
import torchvision.models.quantization as quantize_models


def print_size_of_model(model, label=''):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    print('model: ', label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def load_models(model_name: str):
    assert model_name in ('resnet18', 'resnet50', 'mobilenetv3', 'shufflenetv2')
    
    if model_name == 'resnet18':
        float_model = models.resnet18()
        quantized_model = quantize_models.resnet18(quantize=True)

    elif model_name == 'resnet50':
        float_model = models.resnet50()
        quantized_model = quantize_models.resnet50(quantize=True)

    elif model_name == 'mobilenetv3':
        float_model = models.mobilenetv3_large()
        quantized_model = quantize_models.mobilenet_v3_large(quantize=True)
    
    else: # shufflenetv2
        float_model = models.shufflenet_v2_x0_5()
        quantized_model = quantize_models.shufflenet_v2_x0_5(quantize=True)
    
    return float_model, quantized_model


def comparison_size_of_models(model_name):
    float_model, quantized_model = load_models(model_name)
    f = print_size_of_model(float_model, 'float32')
    q = print_size_of_model(quantized_model, 'int8')
    print("{0:.2f} times smaller".format(f/q))
