import torch

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_graphs(history, project_name):
    train_loss = history['loss']
    valid_loss = history['val_loss']
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(train_loss)), train_loss, label='train')
    plt.plot(np.arange(len(valid_loss)), valid_loss, label='valid')
    plt.legend(loc='best')
    plt.savefig(f'./runs/train/{project_name}/loss_graph.png')
    
    train_acc = history['acc']
    valid_acc = history['val_acc']
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(train_acc)), train_acc, label='train')
    plt.plot(np.arange(len(valid_acc)), valid_acc, label='valid')
    plt.legend(loc='best')
    plt.savefig(f'./runs/train/{project_name}/accuracy_graph.png')
    
    
def plot_results(images, labels, outputs, project_name):
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    outputs = torch.cat(outputs, dim=0)

    nod = len(outputs)
    length = nod // 25 + 1

    for i in range(length):
        plt.figure(figsize=(20, 20))
        for j in range(25):
            idx = i * 25 + j
            if idx > nod - 1:
                break
            plt.subplot(5, 5, j+1)
            plt.imshow(images[idx].permute(1,2,0))
            plt.axis('off')
            plt.title(f'label: {labels[idx]}, output: {outputs[idx]}')
        plt.savefig(f'./runs/test/{project_name}/result{i}.png')
