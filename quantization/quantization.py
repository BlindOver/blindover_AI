import torch
import torch.nn as nn


# for quantization aware training
def prepare_qat(model: nn.Module, backend: str='x86'):
    model.train()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    return torch.quantization.prepare_qat(model)


# for post training quantization
def prepare_ptq(model: nn.Module, backend: str='x86'):
    model.eval()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    return torch.quantization.prepare(model)


# after training
def converting_quantization(model: nn.Module):
    model.eval()
    model = model.cpu()
    return torch.quantization.convert(model)