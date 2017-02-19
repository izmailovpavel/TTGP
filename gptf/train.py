import tensorflow as tf
import numpy as np
import os
from sklearn.cluster import KMeans

from input import NUM_FEATURES, get_batch, prepare_data, make_tensor
from gp import squared_dists, SE, GP, mse, r2
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
    x_tr, y_tr, x_te, y_te = prepare_data(mode="numpy")
    print(x_tr.shape)
    means = KMeans(n_clusters=FLAGS.n_inputs, random_state=241)
    means.fit(x_tr)
    inputs = means.cluster_centers_ 
    iter_per_epoch = int(y_tr.shape[0] / FLAGS.batch_size)
    inputs = make_tensor(inputs, 'inputs')
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr')
    x_te = make_tensor(x_te, 'x_te')
    y_te = make_tensor(y_te, 'y_te')
    maxiter = iter_per_epoch * FLAGS.n_epoch
    x_batch, y_batch = get_batch(x_tr, y_tr, FLAGS.batch_size) 
    gp = GP(SE(1., 1., 1.), inputs) 
    elbo, train_op = gp.fit(x_batch, y_batch, x_tr.get_shape()[0], lr=LR)
#    ms_upd  = gp.mu_sigma(x_tr, y_tr)
    pred = gp.predict(x_te)
    r2 = r2(pred, y_te)
    mse = mse(pred, y_te)
    elbo = gp.elbo(x_tr, y_tr)
    dists = gp.cov(x_tr, x_tr)
    init_ms = gp.initialize_mu_sigma(x_tr, y_tr)

    coord = tf.train.Coordinator()
    init = tf.initialize_all_variables()
    feed_dict = {}
    print('starting session')
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

#        elbo_val = sess.run(elbo)
#        print(elbo_val)
        sess.run(init_ms) 
        for i in range(maxiter):
            if not (i % iter_per_epoch):
                print('Epoch', i/iter_per_epoch, ':')
                print('\tw:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), gp.cov.sigma_n.eval())
                r2_val = sess.run(r2)
                print('r_2 on test set:', r2_val)       
            elbo_val, _ = sess.run([elbo, train_op])
            print('Batch elbo', elbo_val)
        r2_val = sess.run(r2, feed_dict=feed_dict)
        print('Final r2:', r2_val)
