# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import os
from PIL import Image, ImageOps
from random import shuffle

from preproc import *


class Model():
    """
    Common ancestor to ML models
    """
    def __init__(self, batch_size=10,
                 train_path='Training_data/', test_path='Test_data/',
                 input_size=[512, 384, 3]):
        """
        Sets up initial parameters for the model to use.
        :param batch_size: int, batch size to use during training/testing
        :param train_path: str, path to the training images folder
        :param test_path: str, path to the testing images folder
        :param input_size: list(int), size of each input sample
        """
        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.input_size = input_size

        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size] + self.input_size,
                                     name='images')
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 4],
                                     name='labels')

        self.model = self.build_model()
        self.__sess = None

    def build_model(self):
        """
        Build the NN model
        :return: the output of the model, logits, clusters, whatever, really.
        """
        print("Be sure to instantiate child classes!")
        raise NotImplementedError()

    def get_data(self, training=True, random=False):
        """
        Generator for samples and corresponding labels
        :param training: bool, if True, training images are yielded; else test images are yielded
        :param random: bool, if True, the dataset is shuffled; else original order is preserved
        :return: image data and the corresponding labels
        """
        path = self.train_path if training else self.test_path
        ext = '_ext' if os.path.exists(path+"labels_ext.txt") else ''  # checks if extended data set labels exist
        if len(ext):
            print('Using extended dataset...')
        with open(path+"labels%s.txt" % ext, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
            if random:
                shuffle(lines)

        img_data = []
        lbl_data = []
        for line in lines:
            img = np.array(Image.open(path + line[0]))
            if img.shape != self.input_size:
                img = sp.imresize(img, self.input_size)
            # img_data.append(normalize_img(img))
            img_data.append(img)
            lbl_data.append([0., 0., 0., 0.])
            lbl_data[-1][int(line[1])] = 1.
            if len(lbl_data) == self.batch_size:
                yield img_data, lbl_data
                img_data = []
                lbl_data = []

    def get_session(self):
        """
        Singleton for the TF Session
        :return: the tensorflow session of the class
        """
        if not self.__sess:
            self.__sess = tf.Session()
        return self.__sess

    def train(self, num_epochs=5):
        """
        Facilitate training down the line
        :param num_epochs: number of whole passes on the dataset to run while training
        """
        print("Be sure to instantiate child classes!")
        raise NotImplementedError()

    def test(self):
        """
        Facilitate testing the trained model down the line
        :return: test results, accuracy, first/second order error, etc. whatever floats your boat
        """
        print("Be sure to instantiate child classes!")
        raise NotImplementedError()


class SimpleCNN(Model):
    """
    Simple CNN to test performance on dataset
    """

    def build_model(self):
        """
        Just a simple spoopy NN to check if the dataset is up to snuff
        :return: poooop
        """

        conv1 = tf.layers.conv2d(inputs=self.images,
                                 filters=48,
                                 kernel_size=[7, 7],
                                 strides=[3, 3],
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=[2, 2])

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=48,
                                 kernel_size=[5, 5],
                                 strides=[3, 3],
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=[2, 2])

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=[2, 2],
                                        strides=[2, 2])

        # Reshaping the outputs before passing it to the dense layer
        dense1 = tf.layers.dense(tf.reshape(pool3, [self.batch_size, pool3.shape[1]*pool3.shape[2]*pool3.shape[3]]),
                                 800, activation=tf.nn.relu)
        drop1 = tf.layers.dropout(dense1, rate=0.4)

        dense2 = tf.layers.dense(drop1, 600, activation=tf.nn.relu)
        drop2 = tf.layers.dropout(dense2, rate=0.4)

        dense3 = tf.layers.dense(drop2, 400, activation=tf.nn.relu)

        output = tf.layers.dense(dense3, units=self.num_classes, activation=tf.nn.softmax)

        return output

    def train(self, num_epochs=5):
        """
        Trains the neural network for the given number of epochs
        :param num_epochs: the amount of full passes on the training set you want
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.model))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(num_epochs):
                print('Starting epoch #%d' % (e+1))
                b = 1.
                l = 0.
                for batch, label in self.get_data(training=True, random=True):
                    _, batch_loss = sess.run([train_op, loss],
                                    feed_dict={self.images: batch, self.labels: label})
                    l += batch_loss
                    print("Batch #%d, Loss: %f" % (b, l/b))
                    b += 1.

    def test(self):
        return 0.  # Should be a good estimate for this model's performance


class DenseNet(Model):
    """
    DenseNet model.
    @article{Huang2016Densely,
            author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
            title = {Densely Connected Convolutional Networks},
            journal = {arXiv preprint arXiv:1608.06993},
            year = {2016}}
    """
    def __init__(self, batch_size=5,
                 train_path='Training_data_aug/', test_path='Test_data_aug/',
                 input_size=[224, 224, 3],
                 num_blocks=3,
                 L=6,
                 k=32,
                 theta=0.5):
        self.growth_factor = k
        self.compression = theta
        self.blocks = num_blocks
        self.convs = L

        super(DenseNet, self).__init__(batch_size, train_path, test_path, input_size)

    def composite(self, input, _id, kernel_size=3):
        """
        Realizes composite function H_l (batch norm, relu, 3x3 conv)
        :param input: the input of the composite function
        :return: output of H_l(input)
        """
        normie = batch_norm(input, scale=True, fused=True)
        relu = tf.nn.relu(normie)

        kern = tf.get_variable('composite_%d' % _id,
                               shape=[kernel_size, kernel_size, int(relu.shape[-1]), self.growth_factor],
                               initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(relu, kern, strides=[1, 1, 1, 1], padding='SAME')
        # Important to preserve feature map size!!!

        drop = tf.nn.dropout(conv, 0.8)

        return drop

    def transition(self, input, _id):
        """
        Realizes transition layer between dense blocks (batch norm, 1x1 conv, 2x2 avg pool)
        :param input: output of a dense block
        :return: reduced size feature map, ready to be fed to another dense block
        """
        normie = batch_norm(input, scale=True, fused=True)

        kern = tf.get_variable('transition_%d' % _id,
                               shape=[1, 1, int(normie.get_shape()[-1]), int(int(input.get_shape()[-1]) * self.compression)],
                               initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(normie, kern, strides=[1, 1, 1, 1], padding='SAME')
        drop = tf.nn.dropout(conv, 0.8)
        # A 1x1 conv shouldn't reduce featmap size, but still...

        pool = tf.nn.avg_pool(drop, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        return pool

    def bottleneck(self, input, _id):
        """
        Realizes the bottleneck layer to allow (4*growth factor) feature maps through
        Essentially a 1x1 conv with a output featmap size set to (4*growth factor)
        :param input: input to the bottleneck layer
        :return: featuremap with a size reduced to (4*growth factor)
        """
        normie = batch_norm(input, scale=True, fused=True)
        relu = tf.nn.relu(normie)

        kern = tf.get_variable('bottleneck_%d' % _id,
                               shape=[1, 1, int(input.shape[-1]), 4*self.growth_factor],
                               initializer=tf.contrib.layers.xavier_initializer())
        bn = tf.nn.conv2d(relu, kern,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
        drop = tf.nn.dropout(bn, 0.8)

        # print('Bottleneck layer id %d, shape:' % _id, drop.shape)
        return drop

    def internal_layer(self, input, _id):
        """
        Combines bottleneck and composite layers, concatenates their output to the input to grow the feature map
        :param input: the input to append output to
        :return: the extended feature map
        """
        bn = self.bottleneck(input, _id)
        comp = self.composite(bn, _id)
        output = tf.concat(values=(input, comp), axis=3)
        # print('internal layer id: %d, shape:' % _id, output.shape)
        return output

    def dense_block(self, input, _id):  # Is this even good???
        with tf.variable_scope('dense_block_%d' % _id):
            output = self.internal_layer(input, 0)
            for c in range(self.convs - 1):
                output = self.internal_layer(output, c + 1)

        return output

    def output(self, input):
        """
        The final layer of the network. (Avg pool, softmax)
        :param input: the output of the final dense block
        :return: the 4-class classification probabilities
        """
        featmap_size_x = int(input.get_shape()[-3])
        featmap_size_y = int(input.get_shape()[-2])
        avgpool = tf.nn.avg_pool(input, [1, featmap_size_x, featmap_size_y, 1],
                                 strides=[1, featmap_size_x, featmap_size_y, 1],
                                 padding='VALID')
        output = tf.reshape(avgpool, [self.batch_size, avgpool.get_shape()[-1]])
        output = tf.layers.dense(output, units=4)

        return output

    def build_model(self):
        """
        Connects the building blocks defined above into a nice DenseNet :)
        :return: the class probabilities of the network
        """
        print('Setting up network...')
        network = tf.layers.conv2d(self.images, filters=2*self.growth_factor,
                                   kernel_size=7, strides=(2, 2))
        network = tf.layers.average_pooling2d(network, pool_size=(3, 3),
                                              strides=(2, 2))

        for b in range(self.blocks-1):
            network = self.dense_block(network, b)
            network = self.transition(network, b)

        network = self.dense_block(network, self.blocks-1)
        network = self.output(network)

        return network

    def train(self, num_epochs=5):
        self.decay = 0.0001
        self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model)) \
                    + (self.l2 * self.decay)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
        train_op = self.optimizer.minimize(self.loss)

        test_interval = 5

        tf.summary.scalar('Softmax Cross Entropy Loss', self.loss)
        tf.summary.scalar('L2 Loss', self.l2)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train_logs/', self.get_session().graph)

        self.get_session().run(tf.global_variables_initializer())
        for e in range(num_epochs):
            print('Starting epoch #%d' % (e+1))

            b = 1.
            l = 0.
            for batch, label in self.get_data(training=True, random=True):
                _, batch_loss, summary = self.get_session().run([train_op, self.loss, merged],
                                                                feed_dict={self.images: batch, self.labels: label})
                train_writer.add_summary(summary)
                l += batch_loss
                print("Batch #%d, Loss: %f" % (b, batch_loss))
                b += 1.
            print('Ending epoch #%d, average loss: %f' % (e+1, (l/b)))

            if e % test_interval == 0:
                self.test()

        self.test()
        self.get_session().close()
        train_writer.close()

    def test(self):
        """
        Check the accuracy of the network
        :return: the accuracy of the network
        """
        tb = 0.
        acc = 0.
        print('Testing...')
        for batch, label in self.get_data(training=False, random=False):
            output = self.get_session().run(tf.nn.softmax(self.model),
                                          feed_dict={self.images: batch, self.labels: label})

            for o, l in zip(output, label):
                # print(o, l, np.argmax(o), np.argmax(l), (np.argmax(o) == np.argmax(l)))
                acc += (np.argmax(o) == np.argmax(l))
            tb += self.batch_size
        print('Hits: %d, Total:%d, Accuracy: %f' % (acc, tb, acc / tb))
        return acc/tb


class DenseSelu(DenseNet):
    """
    Densenet, but with selu instead of batch norm + relu
    """

    def __init__(self, batch_size=5,
                 train_path='Training_data/', test_path='Test_data/',
                 input_size=[224, 224, 3],
                 num_blocks=3,
                 L=6,
                 k=32,
                 theta=0.5):
        super(DenseSelu, self).__init__(batch_size, train_path, test_path, input_size,
                                        num_blocks, L, k, theta)

    def composite(self, input, _id, kernel_size=3):
        kern = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, int(input.shape[-1]), self.growth_factor],
                                            stddev=tf.sqrt(1/kernel_size*kernel_size)),
                           name='composite_%d' % _id)
        conv = tf.nn.conv2d(input, kern, strides=[1, 1, 1, 1], padding='SAME')
        selu = tf.nn.selu(conv)
        # Important to preserve feature map size!!!

        drop = tf.nn.dropout(selu, 0.8)

        return drop

    def bottleneck(self, input, _id):
        kern = tf.Variable(tf.truncated_normal([1, 1, int(input.shape[-1]), 4 * self.growth_factor],
                                            stddev=1),
                           name='bottleneck_%d' % _id)
        bn = tf.nn.conv2d(input, kern,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
        selu = tf.nn.selu(bn)
        drop = tf.nn.dropout(selu, 0.8)

        # print('Bottleneck layer id %d, shape:' % _id, drop.shape)
        return drop

    def transition(self, input, _id):
        """
        Realizes transition layer between dense blocks (batch norm, 1x1 conv, 2x2 avg pool)
        :param input: output of a dense block
        :return: reduced size feature map, ready to be fed to another dense block
        """
        normie = batch_norm(input, scale=True, fused=True)

        kern = tf.Variable(tf.truncated_normal([1, 1, int(input.get_shape()[-1]),
                                             int(int(input.get_shape()[-1]) * self.compression)],
                                            stddev=1),
                           name='transition_%d' % _id)
        conv = tf.nn.conv2d(normie, kern, strides=[1, 1, 1, 1], padding='SAME')

        drop = tf.nn.dropout(conv, 0.8)
        # A 1x1 conv shouldn't reduce featmap size, but still...

        pool = tf.nn.avg_pool(drop, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        return pool


if __name__ == '__main__':
    print("Please use this as an import instead of running it directly")
    print("Run train_nn.py instead :)")
