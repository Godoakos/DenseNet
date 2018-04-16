# -*- encoding:utf-8 -*-

from models import DenseNet

if __name__ == '__main__':
    cnn = DenseNet(num_blocks=1)
    cnn.train()
