import os
import argparse
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.dataset import load_dataloader
from utils.callback import CheckPoint, EarlyStopping


logger = logging.getLogger('The logs of model training')
logger.setLvel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


def train_on_batch(
    model,
    train_loader,
    device,
    optimizer,
    loss_func,
    log_step,
):
    model.train()
    batch_loss, batch_acc = 0, 0

    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index == labels).sum() / len(outputs)
        
        loss.backward()
        optimizer.step()

        if log_step > 0:
            if (batch + 1) / log_step == 0:
                logger(f'\n[Batch {batch+1}/{len(train_loader)}]'
                       f'  train loss: {loss:.3f}  accuracy: {acc:.3f}')
        
        batch_loss += loss.item()
        batch_acc += acc.item()

    return batch_loss/(batch+1), batch_acc/(batch+1)
    
@torch.no_grad()
def valid_on_batch(
    model,
    valid_loader,
    loss_func,
    device,
    log_step,
):
    model.eval()
    batch_loss, batch_acc = 0, 0
    for batch, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index == labels).sum() / (len(outputs))

        if log_step > 0:
            if (batch + 1) / log_step == 0:
                logger(f'\n[Batch {batch+1}/{len(valid_loader)}]'
                       f'  valid loss: {loss:.3f}  accuracy: {acc:.3f}')

        batch_loss += loss.item()
        batch_acc += acc.item()

    return batch_loss/(batch+1), batch_acc/(batch+1)


def training(
    model: nn.Module,
    train_loader,
    valid_loader,
    lr: float,
    weight_decay: float,
    epochs: int,
    momentum: Optional[float]=0.9,
    optimizer: str='momentum',
    lr_scheduling: bool,
    lr_scheduler,
    check_point: bool=True,
    early_stop: bool=False,
    project_name: str='experiment1',
    class_weight: Optional[torch.Tensor]=None,
    train_log_step: int=300,
    valid_log_step: int = 50,
    es_patience: int=30,
):
    # settings for training
    assert optimizer in ('momentum', 'adam'), \
        f'{optimizer} does not exists.'

    os.makedirs(f'./runs/{project_name}/weights', parents=True, exist_ok=True)
    cp = CheckPoint(verbose=True)

    es_path = f'./runs/{project_name}/weights/es_weight.pt'
    es = EarlyStopping(verbose=True, patience=es_patience, path=es_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device is {device}')
    
    model = model.to(device)
    logger.info('model loading ready.')

    loss_func = nn.CrossEntropyLoss(weight=class_weight)

    if optimizer == 'momentum':
        assert momentum is not None, 
            f'You should set momentum constant.'
        optimizer = optim.SGD(
            model.parameters(),
            momentum=momentum,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        if type(momentum) is float:
            betas = (momentum, 0.999)
        else:
            betas = (0.9, 0.999)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    logger.info(f'optimizer {optimizer} ready.')

    writer = SummaryWriter(log_dir=project_name)

    loss_list, acc_list = [], []
    val_loss_list, val_acc_list = [], []
    start_training = time.time()    
    pbar = tqdm(range(epochs), total=int(epochs))
    for epoch in pbar:
        epoch_time = time.time()

        ##################### training #####################
        train_loss, train_acc = train_on_batch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_func=loss_func,
            log_step=train_log_step,
        )
        ####################################################

        #################### validating ####################
        valid_loss, valid_acc = valid_on_batch(
            model=model,
            valid_loader=valid_loader,
            loss_func=loss_func,
            device=device,
            log_step=valid_log_step,
        )
        ####################################################

        logger(f'\n{"="*30} Epoch {epoch+1}/{epochs} {"="*30}'
               f'\ntime: {time.time() - init_time:.2f}s'
               f'   lr = {optimizer.param_groups[0]["lr"]}')
        logger(f'\ntrain average loss: {train_loss:.3f}'
               f'  accuracy: {train_acc:.3f}')
        logger(f'\nvalid average loss: {valid_loss:.3f}'
               f'  accuracy: {valid_acc:.3f}')
        logger(f'\n{"="*80}')

        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('train/loss', train_loss.item(), epoch)
        writer.add_scalar('train/accuracy', train_acc.item(), epoch)
        writer.add_scalar('valid/loss', valid_loss.item(), epoch)
        writer.add_scalar('valid/accuracy', valid_acc.item(), epoch)

        if lr_scheduling:
            lr_scheduler.step()

        if check_point:
            path = './runs/{}/weights/check_point_{:03d}.pt'.format(project_name, epoch)
            best_path = f'./runs/{project_name}/weights/best.pt'
            cp(valid_loss, model, path)
            cp(valid_loss, model, best_path, save_best=True)

        if early_stop:
            es(valid_loss, model)
            if es.early_stop:
                print('\n##########################\n'
                      '##### Early Stopping #####\n'
                      '##########################')
                break
            
        logger(f'\nTotal training time is {time.time() - start_training:.2f}s')

        return {
            'model': model,
            'loss': loss_list,
            'acc': acc_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
        }


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Model', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--name', type=str, default='experiment1',
                        help='create a new folder')
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['mobilenet', 'shufflenet', 'mnasnet', 'efficientnet'],
                        help='classification model name')
    parser.add_argument('--pretrained', action='store_true',
                        help='load pretrained model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size before applying cropping')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers in cpu')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch Size for training model')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay of optimizer SGD and Adam')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epochs for training model')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum constant for SGD momentum and Adam (beta1)')
    parser.add_argument('--optimizer', default='momentum', type=str,
                        help='set optimizer (sgd momentum and adam)')
    parser.add_argument('--num_classes', default=100, type=int,
                        help='class number of dataset')
    parser.add_argument('--lr_scheduling', action='store_true'
                        help='apply learning rate scheduler')
    parser.add_argument('--check_point', action='store_true'
                        help='save weight file when achieve the best score in validation phase')
    parser.add_argument('--early_stop', action='store_true'
                        help='set early stopping if loss of valid is increased')
    parser.add_argument('--es_patience', default=20, type=int,
                        help='patience to stop training by early stopping')
    parser.add_argument('--train_log_step', type=int, default=40,
                        help='print log of iteration in training loop')
    parser.add_argument('--valid_log_step', type=int, default=10,
                        help='print log of iteration in validating loop')
    
    return parser


def main(args):

    train_loader = load_dataloader(
        path=args.data_path,
        img_size=args.img_size,
        subset='train',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    valid_loader = load_dataloader(
        path=args.data_path,
        img_size=args.img_size,
        subset='valid',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    if args.model == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=args.num_classes, pre_trained=args.pretrained)

    elif args.model == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=args.num_classes, pre_trained=args.pretrained)

    elif args.model == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=args.num_classes, pre_trained=args.pretrained)

    elif args.model == 'mnasnet':
        from models.mnasnet import MNASNet
        model = MNASNet(num_classes=args.num_classes, pre_trained=args.pretrained)

    else:
        raise ValueError(f'{args.model} does not exists')

    summary(model, (3, args.img_size, args.img_size), device='cpu')

    history = training(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        momentum=args.momentum,
        optimizer=args.optimizer,
        lr_scheduling=args.lr_scheduling,
        lr_scheduler=,
        check_point=args.check_point,
        early_stop=args.early_stop,
        project_name=args.name,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
        es_patience=args.es_patience,
    )

    plot_loss_graphs(history, project_name=args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
