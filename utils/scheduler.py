import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLR(_LRScheduler):

    def __init__(self, optimizer, epochs, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = epochs - warmup_epochs
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * \
                (1 + cos(pi * (self.last_epoch - self.warmup_epochs) / self.cosine_epochs)) / 2) \
                    for base_lr in self.base_lrs]


class PolynomialLRDecay(_LRScheduler):
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=1e-5, power=0.9):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr