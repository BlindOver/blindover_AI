# Build Deep Learning Model for blind individuals  
- Models: [ShuffleNetV2](https://arxiv.org/abs/1807.11164), [MobileNetV3](https://arxiv.org/abs/1905.02244), [MNASNet](https://arxiv.org/abs/1807.11626), [EfficientNetV2](https://arxiv.org/abs/2104.00298)  

### Sample images  
- Coca Cola  

<div align='center'>
    <a href='./'>
        <img src = './images/cola_front.JPG' width=200>
    </a>
    <a href='./'>
        <img src = './images/cola_back.JPG' width=200>
    </a>
    <a href='./'>
        <img src = './images/cola_big.JPG' width=200>
    </a>
</div>

- Sprite Zero  

<div align='center'>
    <a href='./'>
        <img src = './images/sprite_zero_front.JPG' width=200>
    </a>
    <a href='./'>
        <img src = './images/sprite_zero_back.JPG' width=200>
    </a>
    <a href='./'>
        <img src = './images/sprite_zero.JPG' width=200>
    </a>
</div>


### Dataset Directory Guide
```
path : dataset/
├── images
│    ├─ class 1
│        ├─ img1.jpg
│        ├─ ...
│    ├─ class 2
│        ├─ img1.jpg
│        ├─ ...
│    ├─ class 3
│        ├─ img1.jpg
│        ├─ ...
│    ├─ ...
│        ├─ ...
│        ├─ ...
```

### Training
```
python3 train.py --data_path '{dataset directory}' --name 'exp' --model '{the one of 4 models}' --pretrained --img_size 224 --num_workers 8 --batch_size 32 --epochs 100 --optimizer 'momentum' --lr_scheduling --check_point
```

### Testing
- Testing model to evaluate the performance in test set
```
python3 test.py --data_path '{dataset directory}' --model '{the one of 4 models}' --weight './runs/exp/weights/best.pt' --img_size 224 --num_workers 8 --batch_size 32 --num_classes 100
```

### Acknowledge  
- 데이터셋 수집에 도움 주신 분들: 이마트24 용인 명지대점, 하나로마트 오산농협본점