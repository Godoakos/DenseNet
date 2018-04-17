import os
import numpy as np
from PIL import Image, ImageOps

def create_manifest_train(train_path='Training_data/'):
    """
    Just a quick helper function to create a manifest of the available training files for future ease
    :param train_path: the path where the data is located
    """
    # The path suffixes for the corresponding folders
    c0_suffix = 'Normal/'
    c1_suffix = 'InSitu/'
    c2_suffix = 'Benign/'
    c3_suffix = 'Invasive/'

    # Making a list of the folder contents
    c0_list = os.listdir(train_path + c0_suffix)
    c1_list = os.listdir(train_path + c1_suffix)
    c2_list = os.listdir(train_path + c2_suffix)
    c3_list = os.listdir(train_path + c3_suffix)

    # Dumping list contents into file with class appended
    with open(train_path+'labels.txt', 'w') as f:
        for x in c0_list:
            f.write(c0_suffix + x + " 0\n")
        for x in c1_list:
            f.write(c1_suffix + x + " 1\n")
        for x in c2_list:
            f.write(c2_suffix + x + " 2\n")
        for x in c3_list:
            f.write(c3_suffix + x + " 3\n")


def create_manifest_test(test_path='Test_data/'):
    classes = ['Normal', 'In situ', 'Benign', 'Invasive']
    with open(test_path + 'labels_orig.txt', 'r') as f:
        lines = f.readlines()
    with open(test_path + 'labels.txt', 'w') as f:
        for line in lines:
            stripped = line.strip().rsplit("\t")
            f.write(stripped[0] + '.tif %d\n' % (classes.index(stripped[1])))

def normalize_img(img):
    mean = np.mean(img)
    img = np.subtract(img, mean)
    var = np.var(img)
    img /= var
    return img


def augment(path):
    with open(path + '/labels.txt', 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    for line in lines:
        orig = Image.open('%s/%s' % (path, line[0]))
        # Flipping, Mirroring, Both
        ops = list()
        ops.append(ImageOps.flip(orig))
        ops.append(ImageOps.mirror(orig))
        ops.append(ImageOps.mirror(ops[0]))
        # Saving
        for i in range(len(ops)):
            ops[i].save('%s/%s_%d.tif' % (path, line[0][:-4], i+1))

def extend_manifest(path):
    with open(path + '/labels.txt', 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    aug = list()
    print(lines)
    for line in lines:
        for i in range(3):
            aug.append(['%s_%d.tif' % (line[0][:-4], i+1), line[1]])
    with open(path + '/labels_ext.txt', 'w') as f:
        for line in lines+aug:
            f.write('%s %s\n' % (line[0], line[1]))


if __name__ == "__main__":
    print("This is a collection of helper functions you can call if you import this file.")
    augment('Training_data')
    extend_manifest('Training_data')
