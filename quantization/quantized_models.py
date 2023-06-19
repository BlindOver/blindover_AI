import torch
import torch.nn as nn
from torchvision.models import quantization


class QuantizedResNet18(nn.Module):

    def __init__(
        self,
        pre_trained: bool=True,
        quantize: bool=True,
        num_classes: int=100,
    ):
        super(QuantizedResNet18, self).__init__()
        
        self.model = quantization.resnet18(pretrained=pre_trained, quantize=quantize)
        self.model.fc.out_features = num_classes

    def forward(self, x):
        return self.model(x)


class QuantizedResNet50(nn.Module):

    def __init__(
        self,
        pre_trained: bool=True,
        quantize: bool=True,
        num_classes: int=100,
    ):
        super(QuantizedResNet50, self).__init__()
        
        self.model = quantization.resnet50(pretrained=pre_trained, quantize=quantize)
        self.model.fc.out_features = num_classes

    def forward(self, x):
        return self.model(x)


class QuantizedMobileNetV3(nn.Module):

    def __init__(
        self,
        pre_trained: bool=True,
        quantize: bool=True,
        num_classes: int=100,
    ):
        super(QuantizedMobileNetV3, self).__init__()
        
        self.model = quantization.mobilenet_v3_large(pretrained=pre_trained, quantize=quantize)
        self.model.classifier[-1].out_features = num_classes

    def forward(self, x):
        return self.model(x)


class QuantizedShuffleNetV2(nn.Module):

    def __init__(
        self,
        pre_trained: bool=True,
        quantize: bool=True,
        num_classes: int=100,
    ):
        super(QuantizedShuffleNetV2, self).__init__()
        self.model = quantization.shufflenet_v2_x0_5(pretrained=pre_trained, quantize=quantize)
        self.model.fc.out_features = num_classes

    def forward(self, x):
        return self.model(x)