import tensorflow as tf
import numpy as np
import os
import time

from gptt_embed.input import prepare_data, make_tensor
#from gptt_embed.gpc import TTGPC
from gptt_embed.gpc_alt import TTGPC
from gptt_embed import grid
import t3f
from t3f import TensorTrain, TensorTrainBatch
from gptt_embed.misc import accuracy, num_correct


class GPCRunner:
    def __init__(self, data_dir, n_inputs, mu_ranks, covs,
            lr=0.01, n_epoch=15, decay=None, batch_size=None,
            preprocess_op=None, te_preprocess_op=None,
            data_type='numpy', log_dir=None, save_dir=None,
            model_dir=None, load_model=False, print_freq=None,
            num_threads=1):
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
            preprocess_op: preprocessing operation for train dataset
            preprocess_op_te: preprocessing operation for test dataset
            data_type: either 'numpy' or 'svmlight' - type of the data encoding
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
        self.preprocess_op = preprocess_op
        self.te_preprocess_op = te_preprocess_op
        self.data_type = data_type
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.model_dir = model_dir
        self.load_model = load_model
        self.print_freq = print_freq
        self.frequent_print = not (print_freq is None)
        self.num_threads = num_threads

    @staticmethod
    def _init_inputs(d, n_inputs):
        inputs = grid.InputsGrid(d, npoints=n_inputs, left=-1.)
        return inputs

    @staticmethod
    def _get_data(data_dir, data_type):
        x_tr, y_tr, x_te, y_te = prepare_data(data_dir, mode=data_type, 
                                                        target='class')
        x_tr = make_tensor(x_tr, 'x_tr')
        y_tr = make_tensor(y_tr.astype(int), 'y_tr', dtype=tf.int64)
        x_te = make_tensor(x_te, 'x_te')
        y_te = make_tensor(y_te.astype(int), 'y_te', dtype=tf.int64)
        return x_tr, y_tr, x_te, y_te
       
    def _make_batches(self, x_tr, y_tr, batch_size, test=False):
        sample_x, sample_y = tf.train.slice_input_producer([x_tr, y_tr], shuffle=True)
        if (self.preprocess_op is not None) and (not test):
            sample_x = self.preprocess_op(sample_x)
        if (self.te_preprocess_op is not None) and test:
            sample_x = self.te_preprocess_op(sample_x)
        sample = [sample_x, sample_y]
        x_batch, y_batch = tf.train.batch(sample, batch_size, 
                num_threads=self.num_threads, capacity=256+3*batch_size)
        return x_batch, y_batch

    @staticmethod
    def _make_mu_initializers(y_init, d):
        n_init = y_init.get_shape()[0]
        y_init_cores = [tf.cast(tf.reshape(y_init, (-1, 1, 1, 1, 1)), tf.float64)]
        for core_idx in range(d):
            if core_idx > 0:
                y_init_cores += [tf.ones((n_init, 1, 1, 1, 1), dtype=tf.float64)]
        y_init = TensorTrainBatch(y_init_cores)
        return y_init

    def eval(self, sess, correct_on_batch, iter_per_test, n_test):
        # TODO: verify this is valid
        correct = 0
        for i in range(iter_per_test):
            correct += sess.run(correct_on_batch)
        accuracy = correct / n_test
        return accuracy

    def run_experiment(self):
                
        start_compilation = time.time()
        d = self.covs.feature_dim()
        x_tr, y_tr, x_te, y_te = self._get_data(self.data_dir, self.data_type)
        x_batch, y_batch = self._make_batches(x_tr, y_tr, self.batch_size)
#        x_batch, y_batch = tf.random_normal((200, 1728)), tf.random_uniform((200,), 0, 10, dtype=tf.int64) 
        x_te_batch, y_te_batch = self._make_batches(x_te, y_te,
                                                self.batch_size, test=True)
        x_init, y_init = self._make_batches(x_tr, y_tr, self.mu_ranks)
        y_init = self._make_mu_initializers(y_init, d)
        inputs = self._init_inputs(d, self.n_inputs)

        N = y_tr.get_shape()[0].value #number of data
        N_te = y_te.get_shape()[0].value #number of data
        iter_per_epoch = int(N / self.batch_size)
        iter_per_te = int(N_te / self.batch_size)
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
        pred = gp.predict(x_te_batch, test=True)
        correct_te_batch = num_correct(pred, y_te_batch)
#        snll_te_batch = reg_snll(pred, sigmas, y_te_batch)

        # Saving results
        model_params = gp.get_params()
        saver = tf.train.Saver(model_params)
        coord = tf.train.Coordinator()
        data_initializer = tf.variables_initializer([x_tr, y_tr, x_te, y_te])
        init = tf.global_variables_initializer()
    
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        # Main session
        with tf.Session() as sess:
            # Initialization
            #writer = tf.summary.FileWriter(self.log_dir, sess.graph) 
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
            for i in range(maxiter):
                if ((not (i % iter_per_epoch)) or 
                (self.frequent_print and not (i % self.print_freq))):
                    # At the end of every epoch evaluate method on test data
                    if i == 0:
                        print('Compilation took', time.time() - start_compilation)
                    print('Epoch', i/iter_per_epoch, ', lr=', lr.eval(), ':')
                    if i != 0:
                        print('\tEpoch took:', time.time() - start_epoch)
                    
                    accuracy = self.eval(sess, correct_te_batch, iter_per_te, N_te)
                    #writer.flush()
                    print('\taccuracy on test set:', accuracy) 
                    print('\taverage elbo:', batch_elbo / iter_per_epoch)
                    batch_elbo = 0
                    start_epoch = time.time()

                # Training operation
                elbo_summary_val, elbo_val, _, _ = sess.run([elbo_summary, 
                                                          elbo, train_op, update_ops])
                batch_elbo += elbo_val
                if self.frequent_print:
                    print('batch', i % iter_per_epoch)
                
            accuracy = self.eval(sess, correct_te_batch, iter_per_te, N_te)
            print('Final accuracy:', accuracy)
            if not self.save_dir is None:
                model_path = saver.save(sess, self.save_dir)
                print("Model saved in file: %s" % model_path)
                gp.cov.projector.save_weights(sess)
