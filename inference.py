from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Padding


def load_image(path: str, img_size: int=224, fill_color: Tuple[int, int, int]=(0,0,0)):

    transformation = transforms.Compose([
        Padding(fill=fill_color),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(path).convert('RGB')
    img = transformation(img)
    img = img.unsqueeze(dim=0)
    return img


def inference(src: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(src)
        prob = F.softmax(outputs)
        result = torch.argmax(prob, dim=1)
    return result
