import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from t3f import TensorTrainBatch

  
def prepare_data(path, mode='svmlight', target='reg'):
  """
  Load and preprocess the data.
  Args:
    path: Path to the data or data folder;
    mode: 'svmlight' or 'numpy'; the type of data to be loaded (.csv or .npy);
      if `mode == npy`, the data must be stored as `path/x_tr.npy`, 
      `path/x_te.npy`, `path/y_tr.npy`, `path/y_te.npy`; if `mode == svmlight`,
      `path` mast be the path to the `.csv` file containing data.
    target: 'reg' or 'class'; type of target values (regression or 
      classification).
  """
  print('Preparing Data')
  if mode == 'svmlight':
    x_, y_ = load_svmlight_file(path)
    if not isinstance(x_, np.ndarray):
        x_ = x_.toarray()
    if not isinstance(y_, np.ndarray):
        y_ = y_.toarray()
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
  scaler_x = StandardScaler()
  x_tr = scaler_x.fit_transform(x_tr) / 3
  x_te = scaler_x.transform(x_te) / 3

  if target =='reg':
    scaler_y = StandardScaler()
    y_tr = scaler_y.fit_transform(y_tr)
    y_te = scaler_y.transform(y_te)
  elif target == 'class':
    pass
  else:
    raise ValueError("unknown target: " + str(target))
  return x_tr, y_tr, x_te, y_te


def prepare_struct_data(data_dir):
  """Loads and preprocesses data for TTGPstruct.

  Args:
    data_dir: Path to the data folder; the data is assumed
      to be stored as `data_dir/x_tr.npy`, `data_dir/x_te.npy`, 
      `data_dir/y_tr.npy`, `data_dir/y_te.npy`, `data_dir/seq_lens_tr.npy`,
      `data_dir/seq_lens_te.npy`.
  Returns:
    x_tr: `np.ndarray`, features for training data.
    y_tr: `np.ndarray`, labels for training data.
    seq_lens_tr: `np.ndarray`, sequence lengths for training data.
    x_te: `np.ndarray`, features for test data.
    y_te: `np.ndarray`, labels for test data.
    seq_lens_te: `np.ndarray`, sequence lengths for test data.
  """
  x_tr = np.load(data_dir+'x_tr.npy')
  y_tr = np.load(data_dir+'y_tr.npy')
  x_te = np.load(data_dir+'x_te.npy')
  y_te = np.load(data_dir+'y_te.npy')
  seq_lens_tr = np.load(data_dir+'seq_lens_tr.npy')
  seq_lens_te = np.load(data_dir+'seq_lens_te.npy')

  D = x_tr.shape[-1]
  x_tr_flat = x_tr.reshape([-1, D])
  x_te_flat = x_te.reshape([-1, D])
  scaler = StandardScaler()
  x_tr_flat = scaler.fit_transform(x_tr_flat)/3
  x_te_flat = scaler.transform(x_te_flat)/3
  x_tr = x_tr_flat.reshape(x_tr.shape)
  x_te = x_te_flat.reshape(x_te.shape)
  return x_tr, y_tr, seq_lens_tr, x_te, y_te, seq_lens_te

def make_tensor(array, name, dtype=tf.float64):
  """Converts the given numpy array to a `tf.Variable`.
  """
  init = tf.constant(array)
  return tf.Variable(init, name=name, trainable=False, dtype=dtype)
