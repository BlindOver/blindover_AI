# Quantize the weights of model to achieve faster inference speed

- Models: ShuffleNetV2, MobileNetV3, EfficientNetV2, ResNet18 and ResNet50
- Comparison size of models (KB):

|                  | ShuffleNetV2 (x0.5) | MobileNetV3 (large) | EfficientNetV2 | ResNet18 | ResNet50 |
| :--------------: | :-----------------: | :-----------------: | :------: | :------: | :-------: |
|   Normal Model   |       1629.99      |      6325.56      | 81722.74 | 44843.61 |94597.31 |
| Quantized Model |       684.15       |       1954.27       | 23666.48 | 11402.92 | 24540.52 |
| shrinkage factor |        2.38      |        3.24       |   3.45   |   3.93   | 3.85|


### Process Guide for Quantization
- Explanation step by step with simple example codes

<details><summary> <b>First Method: PTQ (Post Training Quantization)</b> </summary>
```
step 1. Load a model that include QuantStub() and DeQuantStub() from torch.quantization
```

```python
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class Model(nn.Module):
    def __init__(self, model):
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

model = Model()
```

```
step 2. Training model or loading pre-trained weight
```
    
```python
# training
training(model, train_loader)

# OR loading pre-trained weight
model.load_state_dict(torch.load('pretrained_weight.pt'))
```

```
step 3. Fusing modules such as nn.Conv2d and nn.BatchNorm2d
```
    
```python
import torch

def fuse_modules(model):
    model = model.cpu()
    model.eval()
    modules = [
        ['conv1', 'bn1'],
    ]

    return torch.quantization.fuse_modules(model, modules)

fused_model = fuse_modules(model)
```
    
```
step 4. Preparing quantization for quantizable model with float32 bit
```

```python
import torch

def prepare_ptq(model, backend='x86'):
    model.train()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    return torch.quantization.prepare(model)

prepared_model = prepare_ptq(fused_model)
```

```
step 5. Data calibrilation
```

```python
import torch

def calibration(model, data_loader, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for image, _ in data_loader:
            image = image.to(device)
            _ = model(image)

calibration(prepared_model)
```

```
step 6. Converting the weight from float32 to uint8 for model and saving the quantized weight
```

```python
import torch

def converting(model):
    model.eval()
    model = model.cpu()
    return torch.quantization.convert(model)

quantized_model = converting(prepared_model)
torch.save(quantized_model, './weights/quantized_weight.pt')
```
</details>

<details><summary> <b>Second Method: QAT (Quantization Aware Training)</b> </summary>

```
step 1. Building float32 model or loading pre-trained weight (This step is the same as the step 1 of the above first method (PTQ))
```

```python
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class Model(nn.Module):
    def __init__(self, model):
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

model = Model()
```

```
step 2. Setting training model for model and assigning to cpu device
```

```python
model.train()
model = model.cpu()
```

```
step 3. Fusing modules of model (This step is the same as the step 3 of the above first method (PTQ))
```

```python
import torch

def fuse_modules(model):
    model = model.cpu()
    model.eval()
    modules = [
        ['conv1', 'bn1'],
    ]

    return torch.quantization.fuse_modules(model, modules)

fused_model = fuse_modules(model)
```

```
step 4. Setting qconfig and preparing quantization
```

```python
import torch

def prepare_qat(model, backend: str='x86'):
    model.train()
    model = model.cpu()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    return torch.quantization.prepare_qat(model)

prepared_model = prepare_qat(fused_model)
```

```
step 5. Training model on GPU device (QAT step)
```

```python
import torch

training(model, train_loader, device=torch.device('cuda'))

```

```
step 6. Converting the weight from float32 to uint8 and saving the quantized weight
```

```python
import torch

def converting(model):
    model.eval()
    model = model.cpu()
    return torch.quantization.convert(model)

quantized_model = converting(model)
torch.save(quantized_model, './weights/quantized_weight.pt')
```


### References

- [PyTorch document](https://pytorch.org/docs/stable/quantization.html)
- [My Repository](https://github.com/Sangh0/Quantization)