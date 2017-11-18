import tensorflow as tf
import numpy as np
import os

from TTGP.covariance import SE_multidim
from TTGP.projectors import Identity
from TTGP.gpc_semi_runner import GPCSemiRunner

with tf.Graph().as_default():
    data_dir = ""
    n_inputs = 20
    mu_ranks = 15
    D = 2 # Number of features
    C = 3 # Number of classes
    projector = Identity(D=D)
    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)
    lr = 0.05
    decay = (50, 0.1)
    n_epoch = 10
    batch_size = 50
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models/model.ckpt'
    model_dir = save_dir
    load_model = True
    
    runner=GPCSemiRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
