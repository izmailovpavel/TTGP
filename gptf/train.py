import tensorflow as tf
import numpy as np
import os
from sklearn.cluster import KMeans

from input import prepare_data, make_tensor, batch_subsample
from gp import SE, GP, r2
import grid
import t3f
import t3f.kronecker as kron
from t3f import TensorTrain, TensorTrainBatch
#from tt_batch import *
import time

def get_data():
    '''Data loading and preprocessing.
    '''
    x_tr, y_tr, x_te, y_te = prepare_data(FLAGS.datadir, mode=FLAGS.datatype)
 
    inputs = grid.InputsGrid(x_tr.shape[1], npoints=FLAGS.n_inputs)
    W = inputs.interpolate_kernel(x_tr)
    W = TensorTrainBatch([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in W])
    W_te = inputs.interpolate_kernel(x_te)
    W_te = TensorTrainBatch([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in W_te])
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr')
    n_init = FLAGS.mu_ranks
    W_init_cores = []
    y_init_cores = [tf.reshape(y_tr[:n_init], (n_init, 1, 1, 1, 1))]
    for core_idx in range(W.ndims()):
        W_init_cores += [W.tt_cores[core_idx][:n_init]]
        if core_idx > 0:
            y_init_cores += [tf.ones((n_init, 1, 1, 1, 1), dtype=tf.float64)]
    y_init = TensorTrainBatch(y_init_cores)
    W_init = TensorTrainBatch(W_init_cores)

    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te')
    return x_tr, y_tr, x_te, y_te, W, W_te, inputs, W_init, y_init


def process_flags():
    # Flags definitions
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('n_epoch', 10, 'Number of training epochs')
    flags.DEFINE_integer('batch_size', 128, 'Batch size')
    flags.DEFINE_string('logdir', 'data', 'Directory for progress logging')
    flags.DEFINE_string('datadir', '', 'Directory containing the data')
    flags.DEFINE_string('datatype', 'numpy', 'Data type — numpy or svmlight')
    flags.DEFINE_bool('refresh_stats', False, 
                      'Deletes old events from logdir if True')
    flags.DEFINE_integer('n_inputs', 50, 'Number of inducing inputs')
    flags.DEFINE_integer('mu_ranks', 5, 'TT-ranks of mu')
    flags.DEFINE_bool('load_mu_sigma', False, 'Loads mu and sigma if True')
    
    if FLAGS.refresh_stats:
        print('Deleting old stats')
        os.system('rm -rf ' + FLAGS.logdir)
    return FLAGS


with tf.Graph().as_default():
    
    FLAGS = process_flags()
    x_tr, y_tr, x_te, y_te, W, W_te, inputs, W_init, y_init = get_data()
    iter_per_epoch = int(y_tr.get_shape()[0].value / FLAGS.batch_size)
    maxiter = iter_per_epoch * FLAGS.n_epoch

    # Batches
    load_mu_sigma = FLAGS.load_mu_sigma
    w_batch, y_batch = batch_subsample(W, FLAGS.batch_size, targets=y_tr)
    gp = GP(SE(.7, .2, .1, load_mu_sigma), inputs, W_init, y_init,
            FLAGS.mu_ranks, load_mu_sigma=load_mu_sigma) 
    sigma_initializer = tf.variables_initializer(gp.sigma_l.tt_cores)

    # train_op and elbo
    elbo, train_op = gp.fit(w_batch,  y_batch, x_tr.get_shape()[0], lr=FLAGS.lr)
    elbo_summary = tf.summary.scalar('elbo_batch', elbo)

    # prediction and r2_score on test data
    pred = gp.predict(W_te)
    r2 = r2(pred, y_te)
    r2_summary = tf.summary.scalar('r2_test', r2)

    # Saving results
    mu, sigma_l = gp.get_mu_sigma_cores()
    coord = tf.train.Coordinator()
    cov_initializer = tf.variables_initializer(gp.cov.get_params())
    data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, y_te])

    init = tf.global_variables_initializer()
    
    # Main session
    with tf.Session() as sess:
        # Initialization
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        sess.run(data_initializer)
        sess.run(cov_initializer)
        sess.run(sigma_initializer)
        #sess.run(mu_initializer)
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
        batch_elbo = 0

        for i in range(maxiter):
            if not (i % iter_per_epoch):
                # At the end of every epoch evaluate method on test data
                print('Epoch', i/iter_per_epoch, ':')
                print('\tparams:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), 
                        gp.cov.sigma_n.eval())
                r2_summary_val, r2_val = sess.run([r2_summary, r2])
                writer.add_summary(r2_summary_val, i/iter_per_epoch)
                print('\tr_2 on test set:', r2_val)       
                print('\taverage elbo:', batch_elbo / iter_per_epoch)
                batch_elbo = 0

            # Training operation
            elbo_summary_val, elbo_val, _ = sess.run([elbo_summary, elbo, train_op])
            batch_elbo += elbo_val
            writer.add_summary(elbo_summary_val, i)
            writer.flush()
        
        r2_val = sess.run(r2)
        print('Final r2:', r2_val)

        mu_cores, sigma_l_cores = sess.run([mu, sigma_l])
        
        # Saving results
        if not load_mu_sigma:
            for i, core in enumerate(mu_cores):
                np.save('mu_core'+str(i), core)
            for i, core in enumerate(sigma_l_cores):
                np.save('sigma_l_core'+str(i), core)
