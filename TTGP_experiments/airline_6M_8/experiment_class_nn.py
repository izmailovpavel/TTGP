import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import os
import time

from gptt_embed.misc import accuracy, num_correct
from gptt_embed.input import prepare_data, make_tensor

class NN:
    
    def __init__(self, H1=1000, H2=1000, H3=500, H4=50, D=8, p=0.5):

        with tf.name_scope('layer_1'):
            self.W1 = self.weight_var('W1', [D, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W2 = self.weight_var('W2', [H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [H2, H3])
            self.b3 = self.bias_var('b3', [H3])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [H3, H4])
            self.b4 = self.bias_var('b4', [H4])
        with tf.name_scope('layer_5'):
            self.W5 = self.weight_var('W5', [H4, 2])
            self.b5 = self.bias_var('b5', [2])

        self.p = p
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

    def predict(self, x, test=False):

        # layer 1
        x_input = tf.cast(x, tf.float32)
        h_preact1 = tf.matmul(x_input, self.W1) + self.b1
        h_1 = tf.nn.relu(h_preact1)

        # layer 2
        h_preact2 = tf.matmul(h_1, self.W2) + self.b2
        h_2 = tf.nn.relu(h_preact2)

        # layer 3
        h_preact3 = tf.matmul(h_2, self.W3) + self.b3
        h_3 = tf.nn.relu(h_preact3)

        # layer 4
        h_preact4 = tf.matmul(h_3, self.W4) + self.b4
        h_4 = tf.nn.relu(h_preact4)

        # layer 5
        prediction = tf.matmul(h_4, self.W5) + self.b5
        self.reuse = True

        return prediction

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        bn_vars = []
        for scope in ["norm_1", "norm_2", "norm_3", "norm_4", "norm_5"]:
            bn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return bn_vars + [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4, self.W5] 
    
    def save_weights(self, sess):
        np.save('models/dnn/W1.npy', sess.run(self.W1))
        np.save('models/dnn/b1.npy', sess.run(self.b1))
        np.save('models/dnn/W2.npy', sess.run(self.W2))
        np.save('models/dnn/b2.npy', sess.run(self.b2))
        np.save('models/dnn/W3.npy', sess.run(self.W3))
        np.save('models/dnn/b3.npy', sess.run(self.b3))
        np.save('models/dnn/W4.npy', sess.run(self.W4))
        np.save('models/dnn/b4.npy', sess.run(self.b4))
        np.save('models/dnn/W5.npy', sess.run(self.W5))

def _make_batches(x, y, batch_size, test=False):
    sample_x, sample_y = tf.train.slice_input_producer([x, y], shuffle=True)
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
    data_dir = "data/"
    n_inputs = 10
    mu_ranks = 10
    C = 2
    net = NN(H1=1000, H2=1000, H3=500, H4=50, p=0.5)
    frequent_print = True

    lr = 1e-2
    decay = (1, 0.2)
    n_epoch = 5
    batch_size = 1000

    # data
    x_tr = np.load('data/x_tr.npy')
    y_tr = np.load('data/y_tr.npy')
    x_te = np.load('data/x_te.npy')
    y_te = np.load('data/y_te.npy')
    y_tr = y_tr.astype(int)
    y_te = y_te.astype(int)
    scaler_x = StandardScaler()
    x_tr = scaler_x.fit_transform(x_tr) / 3
    x_te = scaler_x.transform(x_te) / 3
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr', dtype=tf.int64)
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te', dtype=tf.int64)

    # batches
    x_batch, y_batch = _make_batches(x_tr, y_tr, batch_size)
    x_te_batch, y_te_batch = _make_batches(x_te, y_te, batch_size, test=True)

    # Loss
    y_batch_oh = tf.one_hot(y_batch, 2)
    pred = net.predict(x_batch)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_oh, logits=pred))
    pred_labels = tf.argmax(pred, axis=1) 
    print('pred_labels', pred_labels.get_shape())
    print('y_batch', y_batch.get_shape())
    correct_tr_batch = num_correct(pred_labels, y_batch)

    # Accuracy
    pred_raw = net.predict(x_te_batch, test=True)
    pred_te = tf.argmax(pred_raw, axis=1) 
    correct_te_batch = num_correct(pred_te, y_te_batch)
    print('pred_labels', pred_te.get_shape())
    print('y_batch', y_te_batch.get_shape())

    # Optimization params 
    N_te = y_te.get_shape()[0].value
    iter_per_te = int(N_te / batch_size)
    print('iter_per_te', iter_per_te)
    print('N_te', N_te)

    N = y_tr.get_shape()[0].value
    iter_per_epoch = int(N / batch_size)
    print_freq = int(iter_per_epoch / 10)
    print(print_freq)

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
        num_examples = 0
        batch_correct = 0
        for i in range(maxiter):
            if ((not (i % iter_per_epoch)) or 
                (not (i % print_freq))):
                print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
                if i != 0:
                    print('\tEpoch took:', time.time() - start_epoch)
                    print('\taverage loss:', batch_loss / iter_per_epoch)
                    print('\taverage train accuracy:', batch_correct / num_examples)
                accuracy = eval(sess, correct_te_batch, iter_per_te, N_te)
                print('\ttest_accuracy', accuracy)
                batch_loss = 0
                num_examples = 0
                batch_correct = 0
                start_epoch = time.time()

            loss_val, correct_preds, _, _ = sess.run([loss, correct_tr_batch, 
                                                train_op, update_ops])
            batch_correct += correct_preds
            num_examples += batch_size
            batch_loss += loss_val
            if not (i % 100):
                print('batch', i % iter_per_epoch)

        accuracy = eval(sess, correct_te_batch, iter_per_te, N_te)
        print('Final accuracy on test set:', accuracy) 
        net.save_weights(sess)
