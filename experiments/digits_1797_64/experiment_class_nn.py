import tensorflow as tf
import os
import time
#from sklearn.preprocessing import StandardScaler

from gptt_embed.covariance import SE
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gp_runner import GPRunner
from gptt_embed.misc import accuracy
from gptt_embed.input import prepare_data, make_tensor

data_basedir1 = "/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
data_basedir2 = "/Users/IzmailovPavel/Documents/Education/Projects/GPtf/experiments/"

class NN:
    
    def __init__(self, H1=100, H2=100, d=2, D=64, p=0.3):

        with tf.name_scope('layer_1'):
            self.W1 = self.weight_var('W1', [D, H1])
            self.b1 = self.bias_var('b1', [H1])        
        with tf.name_scope('layer_2'):
            self.W2 = self.weight_var('W2', [H1, H2])
            self.b2 = self.bias_var('b2', [H2])
        with tf.name_scope('layer_3'):
            self.W3 = self.weight_var('W3', [H2, d])
            self.b3 = self.bias_var('b3', [d])
        with tf.name_scope('layer_4'):
            self.W4 = self.weight_var('W4', [d, 10])
            self.b4 = self.bias_var('b4', [10])
        self.p = p
        self.d = d
        
    @staticmethod
    def weight_var(name, shape, W=None, trainable=True):
        if not W is None:
            init = tf.constant(W, dtype=tf.float64)
        else:
            init = tf.orthogonal_initializer()(shape=shape, dtype=tf.float64)
        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)

    @staticmethod
    def bias_var(name, shape, b=None, trainable=True):
        if not b is None:
            init = tf.constant(b, dtype=tf.float64)
        else:
            init = tf.constant(0., shape=shape, dtype=tf.float64) 

        return tf.get_variable(name, initializer=init, 
                                dtype=tf.float64, trainable=trainable)

    def predict(self, x):
        l1 = tf.sigmoid(tf.matmul(x, self.W1) + self.b1)
        l1_d = tf.nn.dropout(l1, self.p)
        l2 = tf.sigmoid(tf.matmul(l1_d, self.W2) + self.b2)
        l2_d = tf.nn.dropout(l2, self.p)
        l3 = tf.sigmoid(tf.matmul(l2_d, self.W3) + self.b3)
#	l3_d = tf.nn.dropout(l3, self.p)
        l4 = tf.matmul(l3, self.W4) + self.b4
        return tf.reshape(l4, [-1, 10])

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, 
                self.b4]

    def out_dim(self):
        return self.d

    def save_weights(self, sess):
        W1, b1, W2, b2, W3, b3, W4 = sess.run(self.get_params())
        np.save('W1.npy', W1)
        np.save('b1.npy', b1)
        np.save('W2.npy', W2)
        np.save('b2.npy', b2)
        np.save('W3.npy', W3)
        np.save('b3.npy', b3)
        np.save('W4.npy', W4)

with tf.Graph().as_default():
    data_dir = "data_class/"
    net = NN(H1=100, H2=100, d=4, p=0.5)
    lr = 1e-2

    # data
    x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode='numpy',
                                                    target='class')
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr', dtype=tf.int64)
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te', dtype=tf.int64)
    pred_te = net.predict(x_te)
    accuracy_te = accuracy(tf.argmax(pred_te, axis=1), y_te)
    
    decay = (100, 0.2)
    n_epoch = 300
    batch_size = 200
    
    N = y_tr.get_shape()[0].value
    iter_per_epoch = int(N / batch_size)
    maxiter = iter_per_epoch * n_epoch
    global_step = tf.Variable(0, trainable=False)

    sample = tf.train.slice_input_producer([x_tr, y_tr])
    x_batch, y_batch = tf.train.batch(sample, batch_size)
    y_batch_oh = tf.one_hot(y_batch, 10)
    pred = net.predict(x_batch)
    print(y_batch.get_shape())
    print(pred.get_shape())
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
