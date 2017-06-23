import numpy as np
import tensorflow as tf
import t3f
from scipy.spatial.distance import cdist

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
    ans_ = gp._unary_complexity_penalty()
    with self.test_session() as sess:
      gp.initialize(sess)
      ans = sess.run(ans_)

      Kmm = sess.run(t3f.full(gp._K_mms()))
      mu = sess.run(t3f.full(gp.mus))
      sigma = sess.run(t3f.full(t3f.tt_tt_matmul(
          gp.sigma_ls, t3f.transpose(gp.sigma_ls))))
      mu_prior = np.zeros_like(mu)
      ans_np = 0
      for i in range(mu.shape[0]):
        ans_np += KL(mu[i], sigma[i], mu_prior[i], Kmm[i])
      self.assertAllClose(-ans_np[0, 0], ans)
  
  def test_pairwise_dists(self):
    """Tests `_compute_pairwise_dists` method.
    """
    max_seq_len = 5
    n_seq = 4
    seq_lens = [2, 4, 3, 2]
    n_dims = 3
    n_labels = 2

    x = np.random.randn(n_seq, max_seq_len, n_dims) / 3
    x = tf.constant(x)

    inputs = grid.InputsGrid(n_dims, left=-1., right=1., npoints=5)
    mu_ranks = 5
    projector = Identity(n_dims)
    cov = SE_multidim(n_labels, .7, .3, .1, projector)
    gp = TTGPstruct(cov, inputs, mu_ranks)

    ans_ = gp._compute_pairwise_dists(x)
    with self.test_session() as sess:
      gp.initialize(sess)
      ans = sess.run(ans_)

      x_np = sess.run(x)
      ans_np = []
      for i in range(x_np.shape[0]):
        ans_np.append(cdist(x_np[i], x_np[i])[None, :, :])
      ans_np = np.vstack(ans_np)**2
      self.assertAllClose(ans_np, ans)

  def test_latent_vars_distribution(self):
    """Tests `_latent_vars_distribution` method.
    """
    max_seq_len = 5
    n_seq = 4
    seq_lens = [2, 4, 3, 2]
    n_dims = 3
    n_labels = 2

    x = np.random.randn(n_seq, max_seq_len, n_dims) / 3
    x = tf.constant(x)

    inputs = grid.InputsGrid(n_dims, left=-1., right=1., npoints=5)
    mu_ranks = 5
    projector = Identity(n_dims)
    cov = SE_multidim(n_labels, .7, .3, .1, projector)
    gp = TTGPstruct(cov, inputs, mu_ranks)

    m_un, S_un, m_bin, S_bin = gp._latent_vars_distribution(x, 
        tf.constant(seq_lens))

    with self.test_session() as sess:
      gp.initialize(sess)
      m_un_tf, S_un_tf, m_bin_tf, S_bin_tf = sess.run([m_un, S_un, m_bin, S_bin])
      
      x_np = sess.run(x)
      mus, Sigma_ls, mu_bin, Sigma_bin_l = sess.run([t3f.full(gp.mus), 
          t3f.full(gp.sigma_ls), gp.bin_mu, gp.bin_sigma_l])

      batch_size, max_len, d = x.get_shape().as_list()
      sequence_mask = tf.sequence_mask(seq_lens, maxlen=max_len)
      indices = tf.cast(tf.where(sequence_mask), tf.int32)
      x_flat = tf.gather_nd(x, indices)

      w = sess.run(t3f.full(gp.inputs.interpolate_on_batch(
          gp.cov.project(x_flat))))
      K_nn = sess.run(gp._K_nns(x))
      K_mm = sess.run(t3f.full(gp._K_mms()))

  # Now let's compute the parameters with numpy

    mus = mus[:, :, 0]
    w = w[:, :, 0]
    m_un_np_flat = mus.dot(w.T)
    m_un_np = np.zeros([n_labels, n_seq, max_seq_len])
    S_un_np = K_nn
    for label in range(n_labels):
      prev_seq_len = 0
      Sigma = Sigma_ls[label].dot(Sigma_ls[label].T)
      for seq, seq_len in enumerate(seq_lens):
        # m_un
        m_un_np[label, seq, :seq_len] = m_un_np_flat[label, 
            prev_seq_len:prev_seq_len+seq_len]
        w_cur = w[prev_seq_len:prev_seq_len + seq_len, :]
        cur_cov = w_cur.dot(Sigma.dot(w_cur.T))
        cur_cov -= w_cur.dot(K_mm[label].dot(w_cur.T))
        S_un_np[label, seq, :seq_len, :seq_len] += cur_cov
        S_un_np[label, seq, seq_len:, :] = 0
        S_un_np[label, seq, :, seq_len:] = 0

        prev_seq_len += seq_len
    
    m_bin_np = mu_bin
    S_bin_np = Sigma_bin_l.dot(Sigma_bin_l.T)

    self.assertAllClose(S_un_tf, S_un_np)
    self.assertAllClose(m_un_tf, m_un_np)
    self.assertAllClose(m_bin_tf, m_bin_np)
    self.assertAllClose(S_bin_tf, S_bin_np)

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
