import tensorflow as tf
import numpy as np
from scipy.linalg import orth
from abc import ABCMeta, abstractmethod


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
        pass
    
    @abstractmethod
    def out_dim(self):
        """Returns the dimensionality of the output space.
        """
        pass


class LinearProjector(FeatureTransformer):
    """Linear Feature Transformer"""

    def __init__(self, P=None, d=None, D=None, trainable=True):
        # Projector matrix
        if P is None:
            P = orth(np.random.normal(size=(D, d))).T
        else:
            d, D = P.shape
        with tf.name_scope('Transform_params'):
            self.P = tf.get_variable('Projection_matrix', [d, D],
                                initializer=tf.constant_initializer(P), 
                                dtype=tf.float64, trainable=trainable)

    def transform(self, x, test=False):
        projected = tf.matmul(x, tf.transpose(self.P))
        projected = tf.minimum(projected, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        sess.run(tf.variables_initializer([self.P]))

    def get_params(self):
        return [self.P]

    def out_dim(self):
        return self.P.get_shape()[0].value

    def save_weights(self, sess):
      np.save('P', sess.run(self.P))

class Identity(FeatureTransformer):
    """Identity transform.

    A non-trainable FeatureTransformer, that returns it's
    argument. To be used when no projection is needed.
    """
    def __init__(self, D):
        """
        Args:
            D: dimensionality of the feature space
        """
        self.D = D

    def transform(self, x, test=False):
#        return x
        projected = tf.minimum(x, 1)
        projected = tf.maximum(projected, -1)
        return projected

    def initialize(self, sess):
        pass

    def get_params(self):
        return []

    def out_dim(self):
        return self.D
