import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from TTGP.covariance import SE
from TTGP.projectors import Identity
from TTGP.gp_runner import GPRunner

with tf.Graph().as_default():
    data_dir = "data/"
    n_inputs = 30
    mu_ranks = 25
    D = 9

    projector = Identity(D=D)
    cov = SE(0.7, 0.2, 0.1, projector)

    lr = 1e-3
    decay = (10, 0.2)
    n_epoch = 30
    batch_size = 200
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models'
    model_dir = None
    load_model = False#True
    
    runner=GPRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model)
    runner.run_experiment()
