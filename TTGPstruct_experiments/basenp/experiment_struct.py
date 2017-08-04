import tensorflow as tf
import numpy as np

from TTGP.covariance import SE_multidim, BinaryKernel
from TTGP.projectors import LinearProjector, Identity
from TTGP.gpstruct_runner import GPStructRunner

with tf.Graph().as_default():
  data_dir = 'data_struct/'
  n_inputs = 10
  mu_ranks = 10
#  D = 6438
#  P = np.load('P.npy')
#  projector = LinearProjector(D=D, d=5, trainable=True)
  D = 5
  projector = Identity(D=D)
  n_labels = 3
  cov = SE_multidim(n_labels, 0.7, 0.2, 0.1, projector)
  bin_cov = BinaryKernel(n_labels, alpha=1.)
  
  lr = 5e-4
  decay = (10, 0.2)
  n_epoch = 30
  batch_size = 100
  log_dir = None
  save_dir = None
  model_dir = save_dir
  load_model = False

  runner = GPStructRunner(data_dir, n_inputs, mu_ranks, cov, bin_cov,
      lr=lr, decay=decay, n_epoch=n_epoch, batch_size=batch_size,
      log_dir=log_dir, save_dir=save_dir,
      model_dir=model_dir, load_model=load_model)
  runner.run_experiment()
