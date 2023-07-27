import torch 
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# the number of trainable parameters: 1.62 M

class MobileNetV3(nn.Module):
    def __init__(
        self,
        num_classes=33,
        pre_trained=True,
    ):
        super(MobileNetV3, self).__init__()
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_trained else None
        )
        self.features = model.features
        self.avgpool = model.avgpool
        
        hidden_dim = model.classifier[-1].in_features

        self.classifier = nn.Sequential(
            model.classifier[0],
            model.classifier[1],
            model.classifier[2],
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, dim=1)
        x = self.classifier(x)
        return x