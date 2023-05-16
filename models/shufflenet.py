import torch 
import torch.nn as nn
import torchvision.models as models

# the number of trainable parameters: 0.44 M

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_classes=100,
        pre_trained=True,
    ):
        super(ShuffleNetV2, self).__init__()
        self.model = models.shufflenet_v2_x0_5(pretrained=pre_trained)
        hidden_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)