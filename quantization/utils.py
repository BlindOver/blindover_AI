import os
import time

import torch
import torchvision.models as models


def print_latency(process, req_return=False):
    start = time.time()
    if req_return:
        result = process
        end = time.time()
        print(end - start: 3.f)
        return result
    else:
        process
        end = time.time()
        print(end - start: 3.f)


def print_size_of_model(model, label=''):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    print('model: ', label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


def comparison_size_of_models(model_name):
    float_model, quantized_model = load_models(model_name)
    f = print_size_of_model(float_model, 'float32')
    q = print_size_of_model(quantized_model, 'int8')
    print("{0:.2f} times smaller".format(f / q))