import numpy as np
import matplotlib.pyplot as plt


def plot_loss_graphs(history, project_name):
    train_loss = history['loss']
    valid_loss = history['val_loss']
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(train_loss)), train_loss, label='train')
    plt.plot(np.arange(len(valid_loss)), valid_loss, label='valid')
    plt.legend(loc='best')
    plt.savefig(f'./runs/{project_name}/loss_graph.png')
    
    train_acc = history['acc']
    valid_acc = history['val_acc']
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(train_acc)), train_acc, label='train')
    plt.plot(np.arange(len(valid_acc)), valid_acc, label='valid')
    plt.legend(loc='best')
    plt.savefig(f'./runs/{project_name}/accuracy_graph.png')