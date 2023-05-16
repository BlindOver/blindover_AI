from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


totensor = transforms.ToTensor()


def load_image(path: str, img_size: int=224):
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = totensor(img)
    img = img.unsqueeze(dim=0)
    return img


def inference(src: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(src)
        prob = F.softmax(outputs)
        result = torch.argmax(prob, dim=1)

    return result