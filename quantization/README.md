# Quantize the weights of model to achieve faster inference speed
- Models: [ShuffleNetV2](https://arxiv.org/abs/1807.11164), [MobileNetV3](https://arxiv.org/abs/1905.02244), [ResNet50 and ResNet18](https://arxiv.org/abs/1512.03385)

- Comparison size of models (KB):  

||ShuffleNetV2 (x0.5)|MobileNetV3 (large)|ResNet18|ResNet50|  
|:---:|:---:|:---:|:---:|:---:|  
|Normal Model|5591.39|22120.23|46827.87|102522.81|  
|Quantized Model|1560.61|5628.04|11833.70|26150.91|  
|shrinkage factor|3.58|3.93|3.96|3.92|

### Training Quantized Model (Static Quantization)
```
python3 train.py --quantization --data_path '{dataset directory}' --name 'exp' --model '{the one of 4 models}' --pretrained --img_size 224 --num_workers 8 --batch_size 32 --epochs 100 --optimizer 'momentum' --lr_scheduling --check_point
```

### References
- [PyTorch document](https://pytorch.org/docs/stable/quantization.html)
- [My Repository](https://github.com/Sangh0/Quantization)
