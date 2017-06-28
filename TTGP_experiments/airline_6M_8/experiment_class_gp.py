import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.layers import batch_norm

from gptt_embed.covariance import SE_multidim
from gptt_embed.projectors import FeatureTransformer, LinearProjector, Identity
from gptt_embed.gpc_runner import GPCRunner

with tf.Graph().as_default():
    data_dir = "data/"
    n_inputs = 20
    mu_ranks = 15
    D = 8
    projector = Identity(D=D)
    C = 2

    cov = SE_multidim(C, 0.7, 0.2, 0.1, projector)

    lr = 5e-3
    decay = (1, 0.2)
    n_epoch = 2
    batch_size = 10000
    data_type = 'numpy'
    log_dir = 'log'
    save_dir = 'models/gpnn_100_100_4.ckpt'
    model_dir = save_dir
    load_model = False#True
    
    runner=GPCRunner(data_dir, n_inputs, mu_ranks, cov,
                lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
                data_type=data_type, log_dir=log_dir, save_dir=save_dir,
                model_dir=model_dir, load_model=load_model, print_freq=50)
    runner.run_experiment()
