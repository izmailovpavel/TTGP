import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

import tensorflow as tf
import os
import time

from gptt_embed.misc import accuracy, num_correct
from gptt_embed.input import prepare_data, make_tensor

HEIGHT, WIDTH = 24, 24

class NN():
    
    def __init__(self, Hc1=64, Hc2=64, Hc3=128, Hc4=128, 
            Hc5=128, Hc6=128, Hd1=512, Hd2=512, C=10):

        # Convolutional
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_var('W_conv1', [3, 3, 3, Hc1])
            self.b_conv1 = self.bias_var('b_conv1', [Hc1])        
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_var('W_conv2', [3, 3, Hc1, Hc2])
            self.b_conv2 = self.bias_var('b_conv2', [Hc2])
        with tf.name_scope('conv3'):
            self.W_conv3 = self.weight_var('W_conv3', [3, 3, Hc2, Hc3])
            self.b_conv3 = self.bias_var('b_conv3', [Hc3])        
        with tf.name_scope('conv4'):
            self.W_conv4 = self.weight_var('W_conv4', [3, 3, Hc3, Hc4])
            self.b_conv4 = self.bias_var('b_conv4', [Hc4])
        with tf.name_scope('conv5'):
            self.W_conv5 = self.weight_var('W_conv5', [3, 3, Hc4, Hc5])
            self.b_conv5 = self.bias_var('b_conv5', [Hc5])
        with tf.name_scope('conv6'):
            self.W_conv6 = self.weight_var('W_conv6', [3, 3, Hc5, Hc6])
            self.b_conv6 = self.bias_var('b_conv6', [Hc6])

        # Fully-connected
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_var('W_fc1', [36 * Hc6, Hd1])
            self.b_fc1 = self.bias_var('b_fc1', [Hd1])
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_var('W_fc2', [Hd1, Hd2])
            self.b_fc2 = self.bias_var('b_fc2', [Hd2])
        with tf.name_scope('fc3'):
            self.W_fc3 = self.weight_var('W_fc3', [Hd2, C])
            self.b_fc3 = self.bias_var('b_fc3', [C])
        
        self.reuse = False
        
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
    def max_pool_3x3(x):
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')

    def predict(self, x, test=False):
        batch_size = x.get_shape()[0].value
        x_image = tf.cast(tf.reshape(x, [-1, HEIGHT, WIDTH, 3]), tf.float32)

        # Convolutional layer 1
        h_preact1 = self.conv2d(x_image, self.W_conv1) + self.b_conv1
        norm1 = batch_norm(h_preact1, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv1")
        h_conv1 = tf.nn.relu(norm1)

        # Convolutional layer 2
        h_preact2 = self.conv2d(h_conv1, self.W_conv2) + self.b_conv2
        norm2 = batch_norm(h_preact2, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv2")
        h_conv2 = tf.nn.relu(norm2)

        # Pooling
        h_pool1 = self.max_pool_3x3(h_conv2)


        # Convolutional layer 3
        h_preact3 = self.conv2d(h_pool1, self.W_conv3) + self.b_conv3
        norm3 = batch_norm(h_preact3, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv3")
        h_conv3 = tf.nn.relu(norm3)

        # Convolutional layer 4
        h_preact4 = self.conv2d(h_conv3, self.W_conv4) + self.b_conv4
        norm4 = batch_norm(h_preact4, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv4")
        h_conv4 = tf.nn.relu(norm4)

        # Pooling
        h_pool2 = self.max_pool_3x3(h_conv4)

        # Convolutional layer 5
        h_preact5 = self.conv2d(h_pool2, self.W_conv5) + self.b_conv5
        norm5 = batch_norm(h_preact5, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv5")
        h_conv5 = tf.nn.relu(norm5)

        # Convolutional layer 6
        h_preact6 = self.conv2d(h_conv5, self.W_conv6) + self.b_conv6
        norm6 = batch_norm(h_preact6, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_conv6")
        h_conv6 = tf.nn.relu(norm6)

        # Dense layer 1
        h_flat = tf.reshape(h_conv6, [batch_size, -1])
        print('h_pol2_flat', h_flat.get_shape())
        h_preact_fc1 = tf.matmul(h_flat, self.W_fc1) + self.b_fc1
        norm_fc1 = batch_norm(h_preact_fc1, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_fc1")
        h_fc1 = tf.nn.relu(norm_fc1)

        # Dense layer 2
        h_preact_fc2 = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        norm_fc2 = batch_norm(h_preact_fc2, decay=0.99, is_training=(not test), 
                           reuse=self.reuse, scope="norm_fc2")
        h_fc2 = tf.nn.relu(norm_fc2)

        # Dense layer 3
        h_fc3 = tf.matmul(h_fc2, self.W_fc3)  + self.b_fc3

        self.reuse = True
        return h_fc3

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_bn_vars(self):
        bn_vars = []
        for scope in ["norm_conv1", "norm_conv2", "norm_conv3", 
                      "norm_conv4", "norm_conv5", "norm_conv6",
                      "norm_fc1", "norm_fc2", "norm_fc3"]:
            bn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return bn_vars


    def get_params(self):
        bn_vars = []
        for scope in ["norm_conv1", "norm_conv2", "norm_conv3", 
                      "norm_conv4", "norm_conv5", "norm_conv6",
                      "norm_fc1", "norm_fc2", "norm_fc3"]:
            bn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return bn_vars + [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                self.W_conv3, self.b_conv3, self.W_conv4, self.b_conv4,
                self.W_conv5, self.b_conv5, self.W_conv6, self.b_conv6,
                self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3] 

    def out_dim(self):
        return self.d

def tr_preprocess_op(img):

    img = tf.reshape(img, [32, 32, 3])
    distorted_image = tf.random_crop(img, [HEIGHT, WIDTH, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    res = tf.image.per_image_standardization(distorted_image)
    res = tf.reshape(res, [-1])
    return res

def te_preprocess_op(img):
    img = tf.reshape(img, [32, 32, 3])
    resized_image = tf.image.resize_image_with_crop_or_pad(img, HEIGHT, WIDTH)
    res = tf.image.per_image_standardization(resized_image)
    res = tf.reshape(res, [-1])
    return res

def _make_batches(x, y, batch_size, test=False):
    sample_x, sample_y = tf.train.slice_input_producer([x, y], shuffle=True)
    if not test:
        sample_x = tr_preprocess_op(sample_x)
    elif test:
        sample_x = te_preprocess_op(sample_x)
    sample = [sample_x, sample_y]
    x_batch, y_batch = tf.train.batch(sample, batch_size)
    return x_batch, y_batch

def eval(sess, correct_on_batch, iter_per_test, n_test):
    correct = 0
    for i in range(iter_per_test):
        correct += sess.run(correct_on_batch)
    accuracy = correct / n_test
    return accuracy

with tf.Graph().as_default():
    
    net = NN(Hc1=128, Hc2=128, Hc3=256, Hc4=256, Hc5=256, Hc6=256,
            Hd1=1536, Hd2=512)

    lr = 1e-2
    decay = (30, 0.2)
    n_epoch = 150
    batch_size = 100

    # data
    data_dir = "data_class/"
    x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode='numpy',
                                                    target='class')
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr', dtype=tf.int64)
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te', dtype=tf.int64)

    # batches
    x_batch, y_batch = _make_batches(x_tr, y_tr, batch_size)
    x_te_batch, y_te_batch = _make_batches(x_te, y_te, batch_size, test=True)

    # Loss
    y_batch_oh = tf.one_hot(y_batch, 10)
    pred = net.predict(x_batch)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_oh, logits=pred))
    pred_labels = tf.argmax(pred, axis=1) 
    correct_tr_batch = num_correct(pred_labels, y_batch)

    # Accuracy
    pred_raw = net.predict(x_te_batch, test=True)
    pred_te = tf.argmax(pred_raw, axis=1) 
    correct_te_batch = num_correct(pred_te, y_te_batch)

    # Optimization params 
    N_te = y_te.get_shape()[0].value
    iter_per_te = int(N_te / batch_size)

    N = y_tr.get_shape()[0].value
    iter_per_epoch = int(N / batch_size)

    maxiter = iter_per_epoch * n_epoch
    global_step = tf.Variable(0, trainable=False)

    steps = iter_per_epoch * decay[0]
    lr = tf.train.exponential_decay(lr, global_step, 
        steps, decay[1], staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

        batch_loss = 0
        batch_correct = 0
        for i in range(maxiter):
            if not (i % iter_per_epoch):
                print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
                if i != 0:
                    print('\tEpoch took:', time.time() - start_epoch)
                print('\taverage loss:', batch_loss / iter_per_epoch)
                print('\taverage train accuracy:', batch_correct / N)
                accuracy = eval(sess, correct_te_batch, iter_per_te, N_te)
                print('\ttest accuracy:', accuracy)
                batch_loss = 0
                batch_correct = 0
                start_epoch = time.time()

            loss_val, correct_preds, _, _ = sess.run([loss, correct_tr_batch, 
                                                train_op, update_ops])
            batch_correct += correct_preds
            batch_loss += loss_val

        accuracy_val = sess.run([accuracy_te])
        print('Final accuracy on test set:', accuracy_val) 
