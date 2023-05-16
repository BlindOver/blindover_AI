import torch
import torch.nn as nn
import torchvision.models as models

# the number of trainable parameters: 512 M

class MNASNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        num_classes=100,
        pre_trained=True,
    ):
        super(MNASNet, self).__init__()
        model = models.mnasnet1_3(pretrained=pre_trained)
        self.features = model.layers
        
        hidden_dim = model.classifier[-1].in_features
        self.classifier = nn.Sequential(
            model.classifier[0],
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3]) # global average pooling
        x = self.classifier(x)
        return x