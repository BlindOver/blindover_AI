import torch 
import torch.nn as nn
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub

# the number of trainable parameters: 1.62 M

class MobileNetV3(nn.Module):
    def __init__(
        self,
        num_classes=33,
        pre_trained=True,
        quantization=False,
    ):
        super(MobileNetV3, self).__init__()
        model = models.mobilenet_v3_small(pretrained=pre_trained)
        self.features = model.features
        self.avgpool = model.avgpool
        
        hidden_dim = model.classifier[-1].in_features

        self.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            nn.Linear(hidden_dim, num_classes),
        )

        self.q = quantization
        if quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        if self.q:
            x = self.dequant(x)
        return x