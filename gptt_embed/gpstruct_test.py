import numpy as np
import tensorflow as tf

from gpstruct import TTGPstruct

class TTGPstructTest(tf.test.TestCase):
  """Tests for TTGPstruct class methods.
  """

  def test_binary_complexity_penalty(self):
    """Tests `_binary_complexity_penalty` method.
    """
    pass
     
  def test_unary_complexity_penalty(self):
    """Tests `_unary_complexity_penalty` method.
    """
    pass
  
  def test_pairwise_dists(self):
    """Tests `_compute_pairwise_dists` method.
    """
    pass

  def test_latent_vars_distribution(self):
    """Tests `_latent_vars_distribution` method.
    """
    pass

def KL(mu1, Sigma1, mu2, Sigma2):
  """Computes the KL-divegence between two Normal distributions.

  Note that this function is meant for testing and doesn't include a constant
  term `d` / 2 in the KL-divergence.
  
  Args:
    mu1: `np.ndarray` of shape `d` x 1; expectation of the first Gaussian. 
    Sigma1: `np.ndarray` of shape `d` x d; covariance of the first Gaussian. 
    mu2: np.ndarray` of shape `d` x 1; expectation of the second Gaussian. 
    Sigma2: `np.ndarray` of shape `d` x d; covariance of the second Gaussian. 

  Returns:
    KL-divergence between the specified distributions without the `d`/2 term.
  """
  KL = np.linalg.slogdet(Sigma2)[1] - np.linalg.slogdet(Sigma1)[1]
  Sigma2_inv = np.linalg.inv(Sigma2)
  KL += np.trace(Sigma2_inv.dot(Sigma1)) 
  KL += (mu2 - mu1).T.dot(Sigma2_inv.dot(mu2 - mu1))
  return KL / 2
