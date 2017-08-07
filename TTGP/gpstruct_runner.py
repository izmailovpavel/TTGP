import tensorflow as tf
import numpy as np
import os
import time
import t3f

from TTGP.io import prepare_struct_data, make_tensor
from TTGP.gpstruct import TTGPstruct
from TTGP import grid
from TTGP.misc import accuracy_struct

class GPStructRunner:
  """
  Experiment runner for TT-GPstruct model.
  """
  def __init__(self, data_dir, n_inputs, mu_ranks, covs, bin_cov,
        lr=0.01, n_epoch=15, decay=None, batch_size=None,
        preprocess_op=None, te_preprocess_op=None,
        log_dir=None, save_dir=None,
        model_dir=None, load_model=False, print_freq=None,
        num_threads=1):
    """Runs the experiments for the given parameters and data.

    Args:
      data_dir: Path to the directory, containing the data; the data is assumed
        to be stored as `data_dir/x_tr.npy`, `data_dir/x_te.npy`, 
        `data_dir/y_tr.npy`, `data_dir/y_te.npy`, `data_dir/seq_lens_tr.npy`,
        `data_dir/seq_lens_te.npy`.
      n_inputs: number of inducing inputs per dimension.
      mu_ranks: tt-ranks of the representation of the variational
        parameter mu (expectations of the process at the inducing inputs).
      cov: An object, representing the (product) kernel.
      bin_cov: Covariance object for binary potentials.
      lr: Learning rate for the optimization method (Adam optimizer is used).
      decay: Learning rate decay for the optimization; must be a tuple 
        (decay_rate, decay_steps) --- see tf.train.exponential_decay. Here
        `decay_steps` is counted in terms of training epochs.
      n_epoch: Number of training epochs for the optimization method.
      batch_size: Batch size for the optimization method.
      preprocess_op: Preprocessing operation for training data instances; this
        is meant to be used for data augmentation.
      preprocess_op_te: Preprocessing operation for testing data data instances.
      log_dir: Path to the directory where the logs will be stored.
      save_dir: Path to the directory where the model should be saved.
      model_dir: Path to the directory, where the model should be restored
          from.
      load_model: A `bool`, indicating wether or not to load the model from 
        `model_dir`.
      print_freq: An `int` indicating how often to evaluate the model in terms
        of batches.
      num_threads: An `int`, the number of threads in the batch generating 
        queue.
    """

    self.data_dir = data_dir 
    self.n_inputs = n_inputs
    self.mu_ranks = mu_ranks
    self.covs = covs
    self.bin_cov = bin_cov
    self.lr = lr
    self.n_epoch = n_epoch
    self.decay = decay
    self.batch_size = batch_size
    self.preprocess_op = preprocess_op
    self.te_preprocess_op = te_preprocess_op
    self.log_dir = log_dir
    self.save_dir = save_dir
    self.model_dir = model_dir
    self.load_model = load_model
    self.print_freq = print_freq
    self.frequent_print = not (print_freq is None)
    self.num_threads = num_threads

  @staticmethod
  def _init_inputs(d, n_inputs):
    """Initializes inducing inputs for the model.
    """
    inputs = grid.InputsGrid(d, npoints=n_inputs, left=-1.)
    return inputs

  @staticmethod
  def _get_data(data_dir):
    """Loads the dataset.
    """
    x_tr, y_tr, seq_lens_tr, x_te, y_te, seq_lens_te = prepare_struct_data(
        data_dir)
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr.astype(int), 'y_tr', dtype=tf.int64)
    x_te = make_tensor(x_te, 'x_te')
    seq_lens_tr = make_tensor(seq_lens_tr.astype(int), 'seq_lens_tr', dtype=tf.int64)
    seq_lens_te = make_tensor(seq_lens_te.astype(int), 'seq_lens_tr', dtype=tf.int64)
    return x_tr, y_tr, seq_lens_tr, x_te, y_te, seq_lens_te
     
  def _make_batches(self, x, y, seq_lens, batch_size, test=False):
    """Generates batches for training.

    Args:
      x: `tf.Tensor` containing features.
      y: `tf.Tensor` containing labels.
      seq_lens: `tf.Tensor` containing sequence lengths.
      batch_size: Batch size.
      test: A `bool` indicating whether to use `preprocess_op` or 
        `te_preprocess_op` for preprocessing.

    Returns:
      x_batch: `tf.Tensor` containing features for the current batch.
      y_batch: `tf.Tensor` containing labels for the current batch.
      seq_len_batch: `tf.Tensor` containg sequence lengths for the current 
        batch.
       
    """
    sample_x, sample_y, sample_lens = tf.train.slice_input_producer(
        [x, y, seq_lens], shuffle=True)
    if (self.preprocess_op is not None) and (not test):
      sample_x = self.preprocess_op(sample_x)
    if (self.te_preprocess_op is not None) and test:
      sample_x = self.te_preprocess_op(sample_x)
    sample = [sample_x, sample_y, sample_lens]
    x_batch, y_batch, seq_len_batch = tf.train.batch(sample, batch_size, 
        num_threads=self.num_threads, capacity=256+3*batch_size)
    return x_batch, y_batch, seq_len_batch

  def run_experiment(self):
    """Run the experiment.
    """
    start_compilation = time.time()
    d = self.covs.feature_dim()
    x_tr, y_tr, seq_lens_tr, x_te, y_te_np, seq_lens_te = self._get_data(
        self.data_dir)
    x_batch, y_batch, seq_batch = self._make_batches(x_tr, y_tr, seq_lens_tr, 
        self.batch_size)

    inputs = self._init_inputs(d, self.n_inputs)

    N = y_tr.get_shape()[0].value 
#    N_te = y_te.get_shape()[0].value 
    iter_per_epoch = int(N / self.batch_size)
    maxiter = iter_per_epoch * self.n_epoch

    gp = TTGPstruct(self.covs, self.bin_cov, inputs, self.mu_ranks) 

    # Training ops
    global_step = tf.Variable(0, trainable=False)
    if self.decay is not None:
      steps = iter_per_epoch * self.decay[0]
      lr = tf.train.exponential_decay(self.lr, global_step, 
                                steps, self.decay[1], staircase=True)
    else:
      lr = tf.Variable(self.lr, trainable=False)
    elbo, train_op = gp.fit(x_batch, y_batch, seq_batch, N, lr, global_step)
#    elbo_summary = tf.summary.scalar('elbo_batch', elbo)

    # Saving results
    model_params = gp.get_params()
    saver = tf.train.Saver(model_params)
    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
  
    data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, seq_lens_tr, 
        seq_lens_te])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
    # Main session
    with tf.Session() as sess:
      # Initialization
      sess.run(data_initializer)
      threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
      gp.initialize(sess)
      sess.run(init)

      if self.load_model:
        print('Restoring the model...')
        saver.restore(sess, self.model_dir)
        print('restored.')

      batch_elbo = 0
      start_epoch = time.time()

      # Training
      for i in range(maxiter):
        if ((not (i % iter_per_epoch)) or 
        (self.frequent_print and not (i % self.print_freq))):
          if i == 0:
            print('Compilation took', time.time() - start_compilation)
          print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
          if i != 0:
            print('\tEpoch took:', time.time() - start_epoch)
          
          # Evaluation ops
          pred = gp.predict(x_te, seq_lens_te, sess)
          accuracy_te = accuracy_struct(pred, y_te_np)
          print('\taccuracy on test set:', accuracy_te) 
          print('\taverage elbo:', batch_elbo / iter_per_epoch)
          batch_elbo = 0
          start_epoch = time.time()

        # Training operation
        elbo_val, _, _ = sess.run([elbo, train_op, update_ops])
        batch_elbo += elbo_val
        
      accuracy_val = sess.run(accuracy_te)
      print('Final accuracy:', accuracy_val)
      if not self.save_dir is None:
        model_path = saver.save(sess, self.save_dir)
        print("Model saved in file: %s" % model_path)
        gp.cov.projector.save_weights(sess)
