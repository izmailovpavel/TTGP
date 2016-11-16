import tensorflow as tf
import numpy as np
import os
from sklearn.cluster import KMeans
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface

from input import NUM_FEATURES, get_batch, prepare_data
from linreg import squared_dists, SE, GP, mse, r2
#from evaluate import evaluate

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
    x_tr, y_tr, x_te, y_te = prepare_data()
    print(x_tr[0, :])
    print(y_tr[0])
    means = KMeans(n_clusters=FLAGS.n_inputs, random_state=241)
    means.fit(x_tr)
    inputs = means.cluster_centers_ 
    x_tr_ph = tf.placeholder(tf.float64, shape=x_tr.shape, name='x_tr')
    y_tr_ph = tf.placeholder(tf.float64, shape=y_tr.shape, name='y_tr')
    x_te_ph = tf.placeholder(tf.float64, shape=x_te.shape, name='x_te')
    y_te_ph = tf.placeholder(tf.float64, shape=y_te.shape, name='y_te')
    inputs_ph = tf.placeholder(tf.float64, shape=inputs.shape, name='inputs')
    gp = GP(SE(1., 5., 1.), inputs_ph) 
    train_op = gp.fit(x_tr_ph, y_tr_ph, lr=LR)
    ms_upd  = gp.mu_sigma(x_tr_ph, y_tr_ph)
    pred = gp.predict(x_te_ph)
    r2 = r2(pred, y_te_ph)
    mse = mse(pred, y_te_ph)
    elbo = gp.elbo(x_tr_ph, y_tr_ph)
    dists = gp.cov(x_tr_ph, x_tr_ph)

    init = tf.initialize_all_variables()
    feed_dict = {x_tr_ph: x_tr, y_tr_ph: y_tr, x_te_ph: x_te, y_te_ph: y_te, inputs_ph: inputs}
    print('starting session')
    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

        elbo_val = sess.run(elbo, feed_dict)
        for i in range(MAXEPOCH):
            print('Iteration', i, ':')
            print('\tw:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), gp.cov.sigma_n.eval())
            print('\tELBO:', elbo_val)
            sess.run(train_op, feed_dict=feed_dict)
            elbo_val = sess.run(elbo, feed_dict=feed_dict)
        sess.run(ms_upd, feed_dict)
        r2_val = sess.run(r2, feed_dict=feed_dict)
        print('Final r2:', r2_val)
