import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface

import linreg
from input import NUM_FEATURES, get_batch, prepare_data
from evaluate import evaluate

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('n_epoch', 10, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_string('logdir', 'data', 'Directory for progress logging')
flags.DEFINE_bool('refresh_stats', False, 'Deletes old events from logdir if True')
flags.DEFINE_bool('stoch', False, 'Wether or not to use stochastic optimization')

MAXEPOCH = FLAGS.n_epoch
BATCH_SIZE = FLAGS.batch_size
LR = FLAGS.lr
TRAIN_DIR = FLAGS.logdir

if FLAGS.refresh_stats:
    print('Deleting old stats')
    os.system('rm -rf '+TRAIN_DIR)

with tf.Graph().as_default():
    x_tr, y_tr, x_te, y_te = prepare_data()
    if FLAGS.stoch:
        x_batch, y_batch = get_batch(x_tr, y_tr, BATCH_SIZE)
        iter_per_epoch = int(y_tr.get_shape()[0].value / BATCH_SIZE)
        maxiter = int(MAXEPOCH * iter_per_epoch)
        print(iter_per_epoch)
        with tf.variable_scope('Weights') as scope:
            predictions = linreg.inference(x_batch)
            scope.reuse_variables()
            eval_op = evaluate(x_te, y_te)
            test_err = tf.summary.scalar('val_loss', eval_op)
        loss = linreg.loss(predictions, y_batch)
        train_op = linreg.train(loss, lr=LR)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
            sess.run(init) 
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
            print(x_tr.get_shape()) 
            print('Training...')
            summary, mse =  sess.run([test_err, eval_op])  
            writer.add_summary(summary)
            print('iteration', 0, ':', mse)
            for i in range(maxiter):
                sess.run(train_op)
                if not (i % iter_per_epoch):
                    summary, mse =  sess.run([test_err, eval_op]) 
                    writer.add_summary(summary)
                    print('iteration', i+1, ':', mse) 
    else:
        with tf.variable_scope("Weights") as scope:
            predictions = linreg.inference(x_tr)
            scope.reuse_variables()
            eval_op = evaluate(x_te, y_te)
        loss = linreg.loss(predictions, y_tr)
        tf.scalar_summary('val_loss', eval_op)
        init = tf.initialize_all_variables()
        merged = tf.merge_all_summaries()
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        
        optimizer = ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': MAXEPOCH})

        with tf.Session() as sess:
            sess.run(init)
            optimizer.minimize(sess)
            summary, mse = sess.run([merged, eval_op])
            print('MSE', mse)



