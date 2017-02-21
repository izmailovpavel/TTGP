import tensorflow as tf
import numpy as np
import os
from sklearn.cluster import KMeans

from input import NUM_FEATURES, get_batch, prepare_data, make_tensor
from gp import squared_dists, SE, GP, mse, r2
import grid
import t3f
import t3f.kronecker as kron
from t3f import TensorTrain
from tt_batch import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('n_epoch', 10, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_string('logdir', 'data', 'Directory for progress logging')
flags.DEFINE_bool('refresh_stats', False, 'Deletes old events from logdir if True')
flags.DEFINE_bool('stoch', False, 'Wether or not to use stochastic optimization')
flags.DEFINE_integer('n_inputs', 50, 'Number of inducing inputs')

MAXEPOCH = FLAGS.n_epoch
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.lr
TRAIN_DIR = FLAGS.logdir

if FLAGS.refresh_stats:
    print('Deleting old stats')
    os.system('rm -rf '+TRAIN_DIR)

with tf.Graph().as_default():
    x_tr, y_tr, x_te, y_te = prepare_data(mode="numpy")
   
    inputs = grid.InputsGrid(x_tr.shape[1], npoints=FLAGS.n_inputs)#.full()
    W = inputs.interpolate_kernel(x_tr)
    W = BatchTTMatrices([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in W])
    W_te = inputs.interpolate_kernel(x_te)
    W_te = BatchTTMatrices([tf.reshape(core, [core.get_shape()[0].value, 1, 
                                           core.get_shape()[1].value, 1, 1])
                                           for core in W_te])

    iter_per_epoch = int(y_tr.shape[0] / FLAGS.batch_size)
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr')
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te')
    maxiter = iter_per_epoch * FLAGS.n_epoch
    w_batch, y_batch = batch_subsample(W, FLAGS.batch_size, targets=y_tr)
    gp = GP(SE(1., 1., .01), inputs) 
    elbo, train_op = gp.fit(w_batch,  y_batch, x_tr.get_shape()[0], lr=LR)
    elbo_summary = tf.summary.scalar('elbo_batch', elbo[0, 0])
    check = gp.check_interpolation(W, x_tr)
    check_2 = gp.check_K_ii()
    #check_3 = gp.check_elbo(w_batch, y_batch)
    pred = gp.predict(W_te)
    r2 = r2(pred, y_te)
    r2_summary = tf.summary.scalar('r2_test', r2)
    mse = mse(pred, y_te)
    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    coord = tf.train.Coordinator()
    init = tf.global_variables_initializer()
    print('starting session')
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
#        print(sess.run(check))
#        print(sess.run(check_2))
#        print(sess.run(check_3))
#        exit(0)

        for i in range(maxiter):
            if not (i % iter_per_epoch):
                print('Epoch', i/iter_per_epoch, ':')
                print('\tw:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), 
                        gp.cov.sigma_n.eval())
                r2_summary_val, r2_val = sess.run([r2_summary, r2])
                writer.add_summary(r2_summary_val, i/iter_per_epoch)
                print('r_2 on test set:', r2_val)       
            elbo_summary_val, elbo_val, _ = sess.run([elbo_summary, elbo, train_op])
            writer.add_summary(elbo_summary_val, i)
            writer.flush()
            # print('Batch elbo', elbo_val)
        r2_val = sess.run(r2)
        print('Final r2:', r2_val)
