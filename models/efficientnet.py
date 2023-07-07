import torch 
import torch.nn as nn
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub

# the number of trainable parameters: 20.3 M

class EfficientNetV2(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pre_trained=False,
        quantization=False,
    ):
        super(EfficientNetV2, self).__init__()
        model = models.efficientnet_v2_s(pretrained=pre_trained)
        self.features = model.features
        self.avgpool = model.avgpool
        
        hidden_dim = model.classifier[-1].in_features
        self.classifier = nn.Sequential(
            model.classifier[0],
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