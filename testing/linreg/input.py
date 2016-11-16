import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
data_name = 'cpu_small'
path = '/Users/IzmailovPavel/Documents/Education/Programming/DataSets/Regression/cpusmall(8192, 12).txt'
NUM_FEATURES = 12

def prepare_data():
    """
    Prepares the dataset
    """
    print('Preparing Data')
    x_, y_ = load_svmlight_file(path)
    if not isinstance(x_, np.ndarray):
        x_ = x_.toarray()
    if not isinstance(y_, np.ndarray):
        y_ = y_.toarray()
    x_, y_ = shuffle(x_, y_, random_state=241)
    train_num = int(x_.shape[0] * 0.8)
    x_tr_ = x_[:train_num, :]
    x_te_ = x_[train_num:, :]
    y_tr_ = y_[:train_num]
    y_te_ = y_[train_num:]
    x_tr_init = tf.constant(x_tr_)
    x_te_init = tf.constant(x_te_)
    y_tr_init = tf.constant(y_tr_)
    y_te_init = tf.constant(y_te_)
    x_tr = tf.Variable(x_tr_init, name='x_tr', trainable=False)
    x_te = tf.Variable(x_te_init, name='x_te', trainable=False)
    y_tr = tf.Variable(y_tr_init, name='y_tr', trainable=False)
    y_te = tf.Variable(y_te_init, name='y_te', trainable=False) 
    return x_tr, y_tr, x_te, y_te

def get_batch(x_tr, y_tr, batch_size):
    """
    Iterator, returning bathces
    """
    print('\tIn get_batch...')
    x_example, y_example = tf.train.slice_input_producer([x_tr, y_tr])
    capacity = 32#3 * batch_size#min_after_dequeue  + 3 * batch_size
    print('\tMaking the batch')
    x_batch, y_batch = tf.train.batch([x_example, y_example], batch_size=batch_size, capacity=capacity)
    return x_batch, y_batch
