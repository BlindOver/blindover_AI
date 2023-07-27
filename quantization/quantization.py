import torch


# for training
def prepare_qat(model, backend: str='x86'):
    model.train()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    return torch.quantization.prepare_qat(model)


def prepare_ptq(model, backend: str = 'x86'):
    model.eval()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    return torch.quantization.prepare(model)


# after training
def converting_quantization(model):
    model.eval()
    model = model.cpu()
    return torch.quantization.convert(model)


# for serving
def serving_quantization(model_name: str, weight: str, num_classes: int=33):
    model_fp32 = load_model(model_name, num_classes, quantization=True)
    model_fp32.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_fp32)
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized