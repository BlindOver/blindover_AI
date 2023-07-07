import torch 
import torch.nn as nn
import torchvision.models as models
from torch.quantization import QuantStub, DeQuantStub

# the number of trainable parameters: 11.19 M (ResNet18)

class ResNet18(nn.Module):
    def __init__(
        self,
        num_classes=33,
        pre_trained=True,
        quantization=False,
    ):
        super(ResNet18, self).__init__()
        model = models.resnet18(pretrained=pre_trained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool
        in_features = model.fc.in_features
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        self.q = quantization
        if quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.q:
            x = self.dequant(x)
        return x


# the number of trainable parameters: 23.58 M (ResNet50)

class ResNet50(nn.Module):

    def __init__(
        self,
        num_classes=33,
        pre_trained=True,
        quantization=False,
    ):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained=pre_trained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool
        in_features = model.fc.in_features
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        self.q = quantization
        if quantization:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        if self.q:
            x = self.dequant(x)
        return x