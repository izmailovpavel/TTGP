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

class DataLoader:
    def __init__(self, data_dir, data_type):
        x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode=data_type)


class GPRunner:
    def __init__(self, data_dir, n_inputs, mu_ranks, cov,
            lr=0.01, n_epoch=15, batch_size=None,
            data_type='numpy', log_dir=None, save_dir=None,
            model_dir=None, load_model=False):
        """Runs the experiments for the given parameters and data.

        This class is designed to run the experiments with tt-gp model.
        
        Args:
            data_dir: path to the directory, containing the data
            n_inputs: number of inducing inputs per dimensionality
            mu_ranks: tt-ranks of the representation of the variational
                parameter mu (expectations of the process at the inducing inputs)
            #TODO: make an abstract base class for covariances
            cov: an object, representing covariance
            lr: learning rate of the optimization method
            n_epoch: number of epochs of the optimization method
            batch_size: batch size of the optimization method
            data_type: either 'numpy' or 'svmlight' — type of the data encoding
            log_dir: path to the directory where the logs will be stored
            save_dir: path to the directory where the model should be saved
            model_dir: path to the directory, where the model should be restored
                from
            load_model: a bool, indicating wether or not to load the model
        """

        self.data_dir = data_dir 
        self.n_inputs = n_inputs
        self.mu_ranks = mu_ranks
        self.cov = cov
        self.lr = lr
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.data_type = data_type
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.model_dir = model_dir
        self.load_model = load_model

    @staticmethod
    def _init_inputs(d, n_inputs):
        inputs = grid.InputsGrid(d, npoints=n_inputs, left=-1.)
        return inputs

    @staticmethod
    def _get_data(data_dir, data_type):
        x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode=data_type)
        x_tr = make_tensor(x_tr, 'x_tr')
        y_tr = make_tensor(y_tr, 'y_tr')
        x_te = make_tensor(x_te, 'x_te')
        y_te = make_tensor(y_te, 'y_te')
        return x_tr, y_tr, x_te, y_te
       
    @staticmethod
    def _make_batches(x_tr, y_tr, batch_size)
        sample = tf.train.slice_input_producer([x_tr, y_tr])
        x_batch, y_batch = tf.train.batch(sample, FLAGS.batch_size)
        return x_batch, y_batch

    @staticmethod
    def _make_mu_initializers(n_init, x_tr, y_tr, n_init)
        x_init = x_tr[:n_init]
        y_init_cores = [tf.reshape(y_tr[:n_init], (n_init, 1, 1, 1, 1))]
        for core_idx in range(d):
            if core_idx > 0:
                y_init_cores += [tf.ones((n_init, 1, 1, 1, 1), dtype=tf.float64)]
        y_init = TensorTrainBatch(y_init_cores)
        return x_init, y_init

    def run_experiment(self):
         
        with tf.Graph().as_default():
            x_tr, y_tr, x_te, y_te = self.get_data(data_dir, data_type)
            x_batch, y_batch = self._make_batches(x_tr, y_tr, self.batch_size)
            x_init, y_init = self._make_mu_initializers(x_tr, y_tr, self.mu_ranks)
            N = y_tr.get_shape()[0].value #number of data
            d = ?
            inputs = self.initialize_inputs(d, self.n_inputs)
            
            iter_per_epoch = int(N / self.batch_size)
            maxiter = iter_per_epoch * FLAGS.n_epoch

            if not self.logdir is None:
                print('Deleting old stats')
                os.system('rm -rf ' + FLAGS.logdir)
    

            gp = GP(self.cov, inputs, x_init, y_init, mu_ranks) 
            
            #TODO: do we need this?
            sigma_initializer = tf.variables_initializer(gp.sigma_l.tt_cores)

            # train_op and elbo
            elbo, train_op = gp.fit(x_batch, y_batch, N, lr=self.lr)
            elbo_summary = tf.summary.scalar('elbo_batch', elbo)

            # prediction and r2_score on test data
            pred = gp.predict(x_te)
            r2 = r2(pred, y_te)
            r2_summary = tf.summary.scalar('r2_test', r2)

            # Saving results
            model_params = gp.get_params()
            saver = tf.train.Saver(model_params)
            coord = tf.train.Coordinator()
            data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, y_te])
            init = tf.global_variables_initializer()
    
            # Main session
            with tf.Session() as sess:
                # Initialization
                writer = tf.summary.FileWriter(self.logdir, sess.graph) 
                sess.run(data_initializer)
                gp.initialize(sess)
                sess.run(init)

                if self.load_model:
                    print('Restoring the model...')
                    saver.restore(sess, self.model_dir)
                    print('restored.')
                threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

                batch_elbo = 0
                start_epoch = time.time()
                for i in range(maxiter):
                    if not (i % iter_per_epoch):
                        # At the end of every epoch evaluate method on test data
                        print('Epoch', i/iter_per_epoch, ':')
                        print('\tparams:', gp.cov.sigma_f.eval(), gp.cov.l.eval(), 
                                gp.cov.sigma_n.eval())
                        if i != 0:
                            print('\tEpoch took:', time.time() - start_epoch)
                        r2_summary_val, r2_val = sess.run([r2_summary, r2])
                        writer.add_summary(r2_summary_val, i/iter_per_epoch)
                        writer.flush()
                        print('\tr_2 on test set:', r2_val)       
                        print('\taverage elbo:', batch_elbo / iter_per_epoch)
                        batch_elbo = 0
                        start_epoch = time.time()

                    # Training operation
                    elbo_summary_val, elbo_val, _ = sess.run([elbo_summary, 
                                                              elbo, train_op])
                    batch_elbo += elbo_val
                    writer.add_summary(elbo_summary_val, i)
                
                r2_val = sess.run(r2)
                print('Final r2:', r2_val)
                if not load_model:
                    model_path = saver.save(sess, self.save_dir)
                    print("Model saved in file: %s" % model_path)
