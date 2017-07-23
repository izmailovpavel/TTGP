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
