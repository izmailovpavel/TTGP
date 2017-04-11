import tensorflow as tf
import numpy as np
import os
import time
from sklearn.cluster import KMeans

from input import prepare_data, make_tensor
from gp import GP
from misc import r2
from covariance import SE
from projectors import LinearProjector
import grid
import t3f
#import t3f.kronecker as kron
from t3f import TensorTrain, TensorTrainBatch

def get_data():
    '''Data loading and preprocessing.
    '''
    x_tr, y_tr, x_te, y_te = prepare_data(FLAGS.datadir, mode=FLAGS.datatype)
    
    d = FLAGS.d
    D = x_tr.shape[1]
    Q, S, V = np.linalg.svd(x_tr.T.dot(x_tr))
    
#    if FLAGS.load_model:
##        P = np.load('temp/P.npy')
#        sigma_n = np.load('temp/sigma_n.npy')
#        sigma_f = np.load('temp/sigma_f.npy')
#        l = np.load('temp/l.npy')
#        print('Loaded cov params')
#    else:
    sigma_f = 0.7
    l = 0.2
    sigma_n = 0.1

    P = np.zeros((d, D))
    P[:, :d] = np.eye(d)
    P = LinearProjector(P)
    cov_params = [sigma_f, l, sigma_n, P]

    inputs = grid.InputsGrid(d, npoints=FLAGS.n_inputs, left=-1.)
    x_tr = make_tensor(x_tr, 'x_tr')
    y_tr = make_tensor(y_tr, 'y_tr')

    sample = tf.train.slice_input_producer([x_tr, y_tr])
    x_batch, y_batch = tf.train.batch(sample, FLAGS.batch_size)
    #W_batch = inputs.interpolate_on_batch(x_batch)

    n_init = FLAGS.mu_ranks
    #W_init = inputs.interpolate_on_batch(x_tr[:n_init])
    x_init = x_tr[:n_init]
    y_init_cores = [tf.reshape(y_tr[:n_init], (n_init, 1, 1, 1, 1))]
    for core_idx in range(d):
        if core_idx > 0:
            y_init_cores += [tf.ones((n_init, 1, 1, 1, 1), dtype=tf.float64)]
    y_init = TensorTrainBatch(y_init_cores)

    x_te = make_tensor(x_te, 'x_te')
    #W_te = inputs.interpolate_on_batch(x_te)
    y_te = make_tensor(y_te, 'y_te')
    return x_batch, y_batch, x_te, y_te, x_init, y_init, x_tr, y_tr, inputs, cov_params


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
    flags.DEFINE_bool('load_model', False, 'Loads mu and sigma if True')
    flags.DEFINE_integer('d', 2, 'Projection dimensionality')
    
    if FLAGS.refresh_stats:
        print('Deleting old stats')
        os.system('rm -rf ' + FLAGS.logdir)
    return FLAGS


with tf.Graph().as_default():
    
    FLAGS = process_flags()
    x_batch, y_batch, x_te, y_te, x_init, y_init, x_tr, y_tr, inputs, cov_params = get_data()
    iter_per_epoch = int(y_tr.get_shape()[0].value / FLAGS.batch_size)
    maxiter = iter_per_epoch * FLAGS.n_epoch

    # Batches
    cov_trainable = True
    load_model = FLAGS.load_model
    #w_batch, y_batch = batch_subsample(W, FLAGS.batch_size, targets=y_tr)
    gp = GP(SE(*(cov_params+[cov_trainable])), inputs, x_init, y_init,
            FLAGS.mu_ranks, load_mu_sigma=False)#load_model) 
    sigma_initializer = tf.variables_initializer(gp.sigma_l.tt_cores)

    # train_op and elbo
    elbo, train_op = gp.fit(x_batch, y_batch, x_tr.get_shape()[0], lr=FLAGS.lr)
    elbo_summary = tf.summary.scalar('elbo_batch', elbo)

    # prediction and r2_score on test data
    pred = gp.predict(x_te)
    r2 = r2(pred, y_te)
    r2_summary = tf.summary.scalar('r2_test', r2)

    # Saving results
    model_params = gp.get_params()
    coord = tf.train.Coordinator()
    data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, y_te])
    saver = tf.train.Saver(model_params)
    init = tf.global_variables_initializer()
    
    # Main session
    with tf.Session() as sess:
        # Initialization
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
         
        sess.run(data_initializer)
        gp.initialize(sess)
        sess.run(init)
        if FLAGS.load_model:
            print('Restoring the model...')
            saver.restore(sess, 'temp/model.ckpt')
            print('restored.')
        print(sess.run(gp.cov.projector.P))
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

        batch_elbo = 0
        start_epoch = time.time()
        for i in range(maxiter):
            if not (i % iter_per_epoch):
                # At the end of every epoch evaluate method on test data
                print('Epoch', i/iter_per_epoch, ':')
                print('\tparams:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), 
                        gp.cov.sigma_n.eval())
#                print('\tP:', gp.cov.P.eval())
                if i != 0:
                    print('\tEpoch took:', time.time() - start_epoch)
                r2_summary_val, r2_val = sess.run([r2_summary, r2])
                writer.add_summary(r2_summary_val, i/iter_per_epoch)
                print('\tr_2 on test set:', r2_val)       
                print('\taverage elbo:', batch_elbo / iter_per_epoch)
                batch_elbo = 0
                start_epoch = time.time()

            # Training operation
            elbo_summary_val, elbo_val, _ = sess.run([elbo_summary, elbo, train_op])
            batch_elbo += elbo_val
            writer.add_summary(elbo_summary_val, i)
            writer.flush()
        
        r2_val = sess.run(r2)
        print('Final r2:', r2_val)
        if not load_model:
            model_path = saver.save(sess, 'temp/model.ckpt')
            print("Model saved in file: %s" % model_path)
#        mu_cores, sigma_l_cores = sess.run([mu, sigma_l])        
        # Saving results
#       if not load_mu_sigma:
#            for i, core in enumerate(mu_cores):
#                np.save('temp/mu_core'+str(i), core)
#            for i, core in enumerate(sigma_l_cores):
#                np.save('temp/sigma_l_core'+str(i), core)
#        
#        np.save('temp/sigma_f', gp.cov.sigma_f.eval())  
#        np.save('temp/sigma_n', gp.cov.sigma_n.eval())  
#        np.save('temp/l', gp.cov.l.eval())  
#        np.save('temp/P', gp.cov.P.eval())
