import tensorflow as tf
import numpy as np
from scipy.linalg import orth
from abc import ABCMeta, abstractmethod

import t3f
import t3f.kronecker as kron
from t3f import ops, TensorTrain, TensorTrainBatch
from tensorflow.contrib.opt import ScipyOptimizerInterface


class FeatureTransformer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform(self, x):
        """Transforms features of x.
        Args:
            x: the feature matrix to be transformed.
        """
        pass
    
    @abstractmethod
    def initialize(self, sess):
        """Initializes the parameters of the model.
        
        Args:
            sess: a `Session` instance
        """
        pass

    @abstractmethod
    def get_params(self):
        """Returns a list of parameters of the model.
        """
#    @abstractmethod
#    def save(self, path, sess):
#        """Saves the model.
#        Args:
#            path: path to the directory, where the model will be saved
#            sess: a `Session` instance
#        """
#        pass
#    
#    @abstractmethod
#    def load(self, path):
#        """loads the model.
#        Args:
#            path: path to the directory, where the model is stored
#        """
#        pass


class LinearProjector:
    """Linear Feature Transformer"""

    def __init__(self, P=None, d=None, D=None, trainable=True):
        # Projector matrix
        if P is None:
            P = orth(np.random.normal(size=(d, D)))
        else:
            d, D = P.shape
        with tf.name_scope('Transform_params'):
            self.P = tf.get_variable('Projection_matrix', [d, D],
                                initializer=tf.constant_initializer(P), 
                                dtype=tf.float64, trainable=trainable)

    def transform(self, x):
        projected = tf.matmul(x, tf.transpose(self.P))
        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer([self.P]))

    def get_params(self):
        return [self.P]
