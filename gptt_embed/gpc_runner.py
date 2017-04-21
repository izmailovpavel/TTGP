import tensorflow as tf
import numpy as np
import os
import time

from gptt_embed.input import prepare_data, make_tensor
from gptt_embed.gpc import TTGPC
from gptt_embed import grid
import t3f
from t3f import TensorTrain, TensorTrainBatch
from gptt_embed.misc import accuracy


class GPCRunner:
    def __init__(self, data_dir, n_inputs, mu_ranks, covs,
            lr=0.01, n_epoch=15, decay=None, batch_size=None,
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
            decay: learning rate decay for the oprimization; must be a tuple 
            (decay_rate, decay_steps) --- see tf.train.exponential_decay. Here
            decay_steps is in terms of training epochs
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
        self.covs = covs
        self.lr = lr
        self.n_epoch = n_epoch
        self.decay = decay
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
        x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode=data_type, 
                                                        target='class')
        x_tr = make_tensor(x_tr, 'x_tr')
        y_tr = make_tensor(y_tr, 'y_tr', dtype=tf.int64)
        x_te = make_tensor(x_te, 'x_te')
        y_te = make_tensor(y_te, 'y_te', dtype=tf.int64)
        return x_tr, y_tr, x_te, y_te
       
    @staticmethod
    def _make_batches(x_tr, y_tr, batch_size):
        sample = tf.train.slice_input_producer([x_tr, y_tr])
        x_batch, y_batch = tf.train.batch(sample, batch_size)
        return x_batch, y_batch

    @staticmethod
    def _make_mu_initializers(x_tr, y_tr, n_init, d):
        x_init = x_tr[:n_init]
        y_init_cores = [tf.cast(tf.reshape(y_tr[:n_init], (n_init, 1, 1, 1, 1)), tf.float64)]
        for core_idx in range(d):
            if core_idx > 0:
                y_init_cores += [tf.ones((n_init, 1, 1, 1, 1), dtype=tf.float64)]
        y_init = TensorTrainBatch(y_init_cores)
        return x_init, y_init

    def run_experiment(self):
            d = self.covs[0].feature_dim()
            x_tr, y_tr, x_te, y_te = self._get_data(self.data_dir, self.data_type)
            x_batch, y_batch = self._make_batches(x_tr, y_tr, self.batch_size)
            x_init, y_init = self._make_mu_initializers(x_tr, y_tr, self.mu_ranks, d)
            N = y_tr.get_shape()[0].value #number of data
            inputs = self._init_inputs(d, self.n_inputs)
            
            iter_per_epoch = int(N / self.batch_size)
            maxiter = iter_per_epoch * self.n_epoch

            if not self.log_dir is None:
                print('Deleting old stats')
                os.system('rm -rf ' + self.log_dir)
    

            gp = TTGPC(self.covs, inputs, x_init, y_init, self.mu_ranks) 
            
            # train_op and elbo
            global_step = tf.Variable(0, trainable=False)
            if self.decay is not None:
                steps = iter_per_epoch * self.decay[0]
                print(steps, 'steps before decay')
                lr = tf.train.exponential_decay(self.lr, global_step, 
                                        steps, self.decay[1], staircase=True)
            else:
                lr = tf.Variable(self.lr, trainable=False)
            elbo, train_op = gp.fit(x_batch, y_batch, N, lr, global_step)
            elbo_summary = tf.summary.scalar('elbo_batch', elbo)

            # prediction and r2_score on test data
            pred = gp.predict(x_te)
            accuracy_te = accuracy(pred, y_te)
            accuracy_summary = tf.summary.scalar('accuracy_test', accuracy_te)

            # Saving results
            model_params = gp.get_params()
            saver = tf.train.Saver(model_params)
            coord = tf.train.Coordinator()
            data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, y_te])
            init = tf.global_variables_initializer()
    
            # Main session
            with tf.Session() as sess:
                # Initialization
                writer = tf.summary.FileWriter(self.log_dir, sess.graph) 
                sess.run(data_initializer)
                gp.initialize(sess)
                threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
                sess.run(init)

                if self.load_model:
                    print('Restoring the model...')
                    saver.restore(sess, self.model_dir)
                    print('restored.')

                batch_elbo = 0
                start_epoch = time.time()
                for i in range(maxiter):
                    if not (i % iter_per_epoch):
                        # At the end of every epoch evaluate method on test data
                        print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
                        if i != 0:
                            print('\tEpoch took:', time.time() - start_epoch)
                        accuracy_summary_val, accuracy_val = sess.run(
                                            [accuracy_summary, accuracy_te])
                        writer.add_summary(accuracy_summary_val, i/iter_per_epoch)
                        writer.flush()
                        print('\taccuracy on test set:', accuracy_val)       
                        print('\taverage elbo:', batch_elbo / iter_per_epoch)
                        batch_elbo = 0
                        start_epoch = time.time()

                    # Training operation
                    elbo_summary_val, elbo_val, _ = sess.run([elbo_summary, 
                                                              elbo, train_op])
                    batch_elbo += elbo_val
                    writer.add_summary(elbo_summary_val, i)
                
                print(sess.run(y_te))
                print(sess.run(pred))
                accuracy_val = sess.run([accuracy_te])
                print('Final accuracy:', accuracy_val)
                if not self.save_dir is None:
                    model_path = saver.save(sess, self.save_dir)
                    print("Model saved in file: %s" % model_path)
#                    gp.cov.projector.save_weights(sess)
