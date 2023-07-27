import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ShuffleNet_V2_X0_5_Weights
from torch.quantization import QuantStub, DeQuantStub

# the number of trainable parameters: 0.44 M

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        num_classes=33,
        pre_trained=True,
        quantize=False,
    ):
        super(ShuffleNetV2, self).__init__()
        self.model = models.shufflenet_v2_x0_5(
            weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pre_trained else None
        )
        hidden_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(hidden_dim, num_classes)

        self.quantize = quantize
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.model(x)
        if self.quantize:
            x = self.dequant(x)
        return x