# Quantize the weights of model to achieve faster inference speed

- Models: ShuffleNetV2, MobileNetV3, EfficientNetV2, ResNet18 and ResNet50
- Comparison size of models (KB):

|                  | ShuffleNetV2 (x0.5) | MobileNetV3 (large) | EfficientNetV2 | ResNet18 | ResNet50 |
| :--------------: | :-----------------: | :-----------------: | :------: | :------: | :-------: |
|   Normal Model   |       1629.99      |      6325.56      | 81722.74 | 44843.61 |94597.31 |
| Quantized Model |       684.15       |       1954.27       | 23666.48 | 11402.92 | 24540.52 |
| shrinkage factor |        2.38      |        3.24       |   3.45   |   3.93   | 3.85|

### References

- [PyTorch document](https://pytorch.org/docs/stable/quantization.html)
- [My Repository](https://github.com/Sangh0/Quantization)
