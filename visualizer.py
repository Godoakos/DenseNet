from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf


def visualize_losses(train_loss, test_loss):
    plt.figure(1)
    plt.plot(train_loss, 'r-')
    plt.plot(test_loss, 'b-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def visualize_tensor(tensor):
    plt.subplot()
    plt.imshow(tensor)
    plt.show()


if __name__ == '__main__':
    print('A suite of visualizing stuff (activations, weights, loss, etc.)')
    img = np.zeros([5, 5], dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = random.random()

    visualize_tensor(img)

