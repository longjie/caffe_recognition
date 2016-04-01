#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from chainer.functions import caffe
import chainer.functions as F

class GoogleNet:
    def __init__(self, xp, model):
        print('Loading Caffe model file %s as googlenet...' % model, file=sys.stderr)
        self.func = caffe.CaffeFunction(model)
        print('Loaded', file=sys.stderr)

        self.in_size = 224
        # Constant mean over spatial pixels
        self.mean_image = xp.ndarray((3, self.in_size, self.in_size), dtype=xp.float32)
        self.mean_image[0] = 104
        self.mean_image[1] = 117
        self.mean_image[2] = 123

    def forward(self, x, t):
        
        y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
                       disable=['loss1/ave_pool', 'loss2/ave_pool'],
                       train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def predict(self, x):
        y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
                       disable=['loss1/ave_pool', 'loss2/ave_pool'],
                       train=False)
        return F.softmax(y)

class AlexNet:
    def __init__(self, model, mean, gpu=0):
        print('Loading Caffe model file %s as googlenet...' % model, file=sys.stderr)
        self.func = caffe.CaffeFunction(model)
        print('Loaded', file=sys.stderr)
        
        in_size = 227
        mean_image = xp.load(mean)

    def forward(self, x, t):
        y, = self.func(inputs={'data': x}, outputs=['fc8'], train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def perdict(self, x):
        y, = self.func(inputs={'data': x}, outputs=['fc8'], train=False)
        return F.softmax(y)

