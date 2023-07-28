from typing import *

import torch
import torch.nn as nn

from .quantization import converting_quantization
from .utils import fuse_modules


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
    model: nn.Module, str,
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
    model = converting_quantization(model)
    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    return model