#!/usr/bin/env python
# coding: utf-8
# Created on Mon Apr 11 16:55:39 2022
# @author: Lu Jian
# Email:janelu@live.cn; lujian@sdc.icbc.com.cn

from paddle.optimizer.lr import LRScheduler

class InverseSqrt(LRScheduler):
    def __init__(self,
                learning_rate,
                warmup_init_lr,
                warmup_updates,
                last_epoch=-1,
                verbose=False):
        self.warmup_updates=warmup_updates
        self.lr_step = (learning_rate - warmup_init_lr) / self.warmup_updates
        self.decay_factor = learning_rate * self.warmup_updates**0.5
        super(InverseSqrt, self).__init__(warmup_init_lr, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <self.warmup_updates:
            return self.base_lr + self.lr_step*self.last_epoch
        return self.decay_factor * self.last_epoch**-0.5