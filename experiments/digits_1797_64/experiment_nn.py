import tensorflow as tf
import os
import time
#from sklearn.preprocessing import StandardScaler

from gptt_embed.gp import GP
from gptt_embed.covariance import SE
from gptt_embed.projectors import FeatureTransformer, LinearProjector
from gptt_embed.gp_runner import GPRunner
from gptt_embed.misc import mse, r2
from gptt_embed.input import prepare_data, make_tensor

data_basedir1 = "/Users/IzmailovPavel/Documents/Education/Programming/DataSets/"
data_basedir2 = "/Users/IzmailovPavel/Documents/Education/Projects/GPtf/experiments/"

class NN:
    
    def __init__(self, H1=100, H2=100, d=2, D=64):

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
            self.W4 = self.weight_var('W4', [d, 1])

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
        l2 = tf.sigmoid(tf.matmul(l1, self.W2) + self.b2)
        l3 = tf.sigmoid(tf.matmul(l2, self.W3) + self.b3)
        l4 = tf.matmul(l3, self.W4)
        return l4

    def initialize(self, sess):
        sess.run(tf.variables_initializer(self.get_params()))

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4]

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
    data_dir = ""
    net = NN(H1=100, H2=100, d=4)
    lr = 1e-2

    # data
    x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode='numpy')
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr')
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te')
    pred = net.predict(x_tr)
    loss = mse(pred, y_tr)
    pred_te = net.predict(x_te)
    r2_te = r2(pred_te, y_te)
    mse_te = mse(pred_te, y_te)
    
#    decay = (500, 0.2)
    n_epoch = 300
    batch_size = 200
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    
    N = y_tr.get_shape()[0].value
    iter_per_epoch = int(N / batch_size)
    maxiter = iter_per_epoch * n_epoch

#    data_type = 'numpy'
#    log_dir = 'log'
#    save_dir = 'models/nn_100_100_4_1.ckpt'
#    model_dir = save_dir
#    load_model = False#True
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        batch_mse = 0
        for i in range(maxiter):
            if not (i % iter_per_epoch):
                print('Epoch', i/iter_per_epoch, ', lr=', lr, ':')
                if i != 0:
                    print('\tEpoch took:', time.time() - start_epoch)
                print('\taverage mse:', batch_mse / iter_per_epoch)
                r2_val, mse_val = sess.run([r2_te, mse_te])
                print('\tr_2 on test set:', r2_val) 
                print('\tmse on test set:', mse_val) 
                batch_mse = 0
                start_epoch = time.time()

            mse, _ = sess.run([loss, train_op])
            batch_mse += mse

        r2_val, mse_val = sess.run([r2_te, mse_te])
        print('Final r_2 on test set:', r2_val) 
        print('Final mse on test set:', mse_val) 
