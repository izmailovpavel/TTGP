import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

import tensorflow as tf
import os
import time

from gptt_embed.misc import accuracy
from gptt_embed.input import prepare_data, make_tensor


class NN:
    
    def __init__(self, H1=32, H2=64, H3=1024, C=10, p=0.5):

        with tf.name_scope('layer_1'):
            self.W_conv1 = self.weight_var('W1', [5, 5, 1, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W_conv2 = self.weight_var('W2', [5, 5, H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [7 * 7 * H2, H3])
            self.b3 = self.bias_var('b3', [H3])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [H3, C])
            self.b4 = self.bias_var('b4', [C])
        
        self.keep_prob = p
        
    @staticmethod
    def weight_var(name, shape, trainable=True):
        init = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float32, trainable=trainable)
    @staticmethod
    def bias_var(name, shape, trainable=True):
        init = tf.constant(0.1, shape=shape, dtype=tf.float32) 
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float32, trainable=trainable)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

    def predict(self, x, test=False):
        batch_size = x.get_shape()[0].value
        x_image = tf.cast(tf.reshape(x, [-1, 28, 28, 1]), tf.float32)

        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [batch_size, -1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W3) + self.b3)
        if not test:
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        else:
            print('test')
            h_fc1_drop = h_fc1

        l4 = tf.matmul(h_fc1_drop, self.W4) + self.b4
        
        return l4


with tf.Graph().as_default():
    
    net = NN(H1=32, H2=64, H3=1024, p=0.5)
    lr = 1e-3

    # data
    data_dir = "data_class/"
    x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode='numpy',
                                                    target='class')
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr', dtype=tf.int64)
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te', dtype=tf.int64)
    pred_te = net.predict(x_te, test=True)
    accuracy_te = accuracy(tf.argmax(pred_te, axis=1), y_te)
    
    decay = (10, 0.2)
    n_epoch = 30
    batch_size = 50
    
    N = y_tr.get_shape()[0].value
    iter_per_epoch = int(N / batch_size)
    maxiter = iter_per_epoch * n_epoch
    global_step = tf.Variable(0, trainable=False)

    sample = tf.train.slice_input_producer([x_tr, y_tr])
    x_batch, y_batch = tf.train.batch(sample, batch_size)
    y_batch_oh = tf.one_hot(y_batch, 10)
    pred = net.predict(x_batch)
    print('y_batch', y_batch.get_shape())
    print('pred', pred.get_shape())
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_oh, logits=pred))

    steps = iter_per_epoch * decay[0]
    lr = tf.train.exponential_decay(lr, global_step, 
        steps, decay[1], staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

        batch_loss = 0
        for i in range(maxiter):
            if not (i % iter_per_epoch):
                print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
                if i != 0:
                    print('\tEpoch took:', time.time() - start_epoch)
                print('\taverage loss:', batch_loss / iter_per_epoch)
                accuracy_val = sess.run([accuracy_te])
                print('\taccuracy on test set:', accuracy_val) 
                batch_loss = 0
                start_epoch = time.time()

            loss_val, _ = sess.run([loss, train_op])
            batch_loss += loss_val

        accuracy_val = sess.run([accuracy_te])
        print('Final accuracy on test set:', accuracy_val) 
