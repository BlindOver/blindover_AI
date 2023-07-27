import torch 
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# the number of trainable parameters: 20.3 M

class EfficientNetV2(nn.Module):
    def __init__(
        self,
        num_classes=100,
        pre_trained=False,
    ):
        super(EfficientNetV2, self).__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.avgpool = model.avgpool
        
        hidden_dim = model.classifier[-1].in_features
        self.classifier = nn.Sequential(
            model.classifier[0],
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x