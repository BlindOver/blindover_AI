# Quantize the weights of model to achieve faster inference speed

- Models: ShuffleNetV2, MobileNetV3, EfficientNetV2, ResNet18 and ResNet50
- Comparison size of models (KB):

|                  | ShuffleNetV2 (x0.5) | MobileNetV3 (large) | EfficientNetV2 | ResNet18 | ResNet50 |
| :--------------: | :-----------------: | :-----------------: | :------: | :------: | :-------: |
|   Normal Model   |       1629.99      |      6325.56      | 81722.74 | 44843.61 |94597.31 |
| Quantized Model |       684.15       |       1954.27       | 23666.48 | 11402.92 | 24540.52 |
| shrinkage factor |        2.38      |        3.24       |   3.45   |   3.93   | 3.85|


### Process Guide for Quantization
- PTQ (Post Training Quantization)
    ```
    
        ```python
        
        ```
    step 1. Load a model that include QuantStub() and DeQuantStub() from torch.quantization
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
        ```
    step 2. training or pre-trained weight 세팅
    step 3. modules fusion
    step 4. prepare quantization
    step 5. data callibrilation
    step 6. convert float32 to uint8
    ```


### References

- [PyTorch document](https://pytorch.org/docs/stable/quantization.html)
- [My Repository](https://github.com/Sangh0/Quantization)
