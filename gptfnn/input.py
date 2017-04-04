import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from t3f import TensorTrainBatch


def prepare_data(path, mode='svmlight'):
    """
    Prepares the dataset
    """
    print('Preparing Data')
    if mode == 'svmlight':
        x_, y_ = load_svmlight_file(path)
        if not isinstance(x_, np.ndarray):
            x_ = x_.toarray()
        if not isinstance(y_, np.ndarray):
            y_ = y_.toarray()
#        x_, y_ = shuffle(x_, y_, random_state=241)
        train_num = int(x_.shape[0] * 0.8)
        x_tr = x_[:train_num, :]
        x_te = x_[train_num:, :]
        y_tr = y_[:train_num]
        y_te = y_[train_num:]
    elif mode == 'numpy':
        x_tr = np.load(path+'x_tr.npy')
        y_tr = np.load(path+'y_tr.npy')
        x_te = np.load(path+'x_te.npy')
        y_te = np.load(path+'y_te.npy')
    else:
        raise ValueError("unknown mode: " + str(mode))
    # TODO: project to a unit cube
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_tr = scaler_x.fit_transform(x_tr) / 3
    x_te = scaler_x.transform(x_te) / 3
    y_tr = scaler_y.fit_transform(y_tr)
    y_te = scaler_y.transform(y_te)
#    x_max = np.max(x_tr, axis=0)
#    x_min = np.min(x_tr, axis=0)
#    x_tr = (x_tr - x_min[None, :]) / (x_max - x_min)[None, :]
#    x_te = (x_te - x_min[None, :]) / (x_max - x_min)[None, :]
#    
#    x_te[x_te < 0] = 0.
#    x_te[x_te > 1] = 1.
    return x_tr, y_tr, x_te, y_te

def make_tensor(array, name):
    init = tf.constant(array)
    return tf.Variable(init, name=name, trainable=False)

