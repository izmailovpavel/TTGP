import numpy as np
import tensorflow as tf
import t3f

from gptt_embed.gpstruct import TTGPstruct
from gptt_embed import grid
from gptt_embed.covariance import SE_multidim
from gptt_embed.projectors import Identity


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
    n_dims = 3
    n_labels = 2
    inputs = grid.InputsGrid(n_dims, left=-1., right=1., npoints=5)
    mu_ranks = 5
    projector = Identity(n_dims)
    cov = SE_multidim(n_labels, .7, .3, .1, projector)
    gp = TTGPstruct(cov, inputs, mu_ranks)
    ans = gp._unary_complexity_penalty()
    with self.test_session() as sess:
      gp.initialize(sess)
      Kmm = sess.run(t3f.full(gp._K_mms()))
      mu = sess.run(t3f.full(gp.mus))
      sigma = sess.run(t3f.full(t3f.tt_tt_matmul(
          gp.sigma_ls, t3f.transpose(gp.sigma_ls))))
      mu_prior = np.zeros_like(mu)
      ans_np = 0
      for i in range(mu.shape[0]):
        ans_np += KL(mu[i], sigma[i], mu_prior[i], Kmm[i])
      self.assertAllClose(-ans_np[0, 0], sess.run(ans))
  
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


if __name__ == "__main__":
  tf.test.main()
