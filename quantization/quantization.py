import torch


def load_model(model_name, num_classes, quantization=True):
    if model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=num_classes, pre_trained=False, quantization=quantization)

    elif model_name == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=num_classes, pre_trained=False, quantization=quantization)

    elif model_name == 'resnet18':
        from models.resnet import ResNet18
        model = ResNet18(num_classes=num_classes, pre_trained=False, quantization=quantization)

    elif model_name == 'resnet50':
        from models.resnet import ResNet50
        model = ResNet50(num_classes=num_classes, pre_trained=False, quantization=quantization)
    
    elif model_name == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=num_classes, pre_trained=False, quantization=quantization)

    else:
        raise ValueError(f'{model_name} does not exists')
    
    return model


# for training
def prepare_quantization(model):
    model.eval()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    return torch.quantization.prepare(model)


# for training
def model_quantization(model):
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