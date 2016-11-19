import tensorflow as tf
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface

from input import NUM_FEATURES, get_batch, prepare_data
from gp import squared_dists, SE, GP, mse, r2
#from evaluate import evaluate

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_iter', 10, 'Number of training epochs')
flags.DEFINE_string('logdir', 'data', 'Directory for progress logging')
flags.DEFINE_bool('refresh_stats', False, 'Deletes old events from logdir if True')
flags.DEFINE_integer('n_inputs', 50, 'Number of inducing inputs')

TRAIN_DIR = FLAGS.logdir

if FLAGS.refresh_stats:
    print('Deleting old stats')
    os.system('rm -rf '+TRAIN_DIR)

with tf.Graph().as_default():
    x_tr, y_tr, x_te, y_te = prepare_data()
    means = KMeans(n_clusters=FLAGS.n_inputs, random_state=241)
    means.fit(x_tr)
    inputs = means.cluster_centers_ 
    w = np.array([1., 10., .5])
    #w = np.array([1, 5, 1])
    gp = GP(SE(), w, inputs) 
    
    init = tf.initialize_all_variables()
    print('starting session')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
        print(FLAGS.n_iter)
        gp.fit(x_tr, y_tr, FLAGS.n_iter, sess)
        preds = gp.predict(x_te, sess)

print('Final r2:', r2_score(y_te, preds))
