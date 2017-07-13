import numpy as np
import tensorflow as tf
import t3f

from gptt_embed import misc

class MiscTest(tf.test.TestCase):
  """Tests functions from `misc` module.
  """

  def test_kron_sequence_pairwise_quadratic_form(self):
    """Tests `_kron_sequence_pairwise_quadratic_form` function.
    """
    # Matrix batch
    kron_mat_init = t3f.initializers.random_matrix_batch(((6, 7), (6, 7)), 
        tt_rank=1, batch_size=5)
    kron_mat = t3f.get_variable('kron_mat', initializer=kron_mat_init)

    # Vector batch
    seq_lens, sum_len, max_len = tf.constant([2, 4, 3]), 9, tf.constant(6)
    kron_vec_init = t3f.initializers.random_matrix_batch(((6, 7), (1, 1)), 
        tt_rank=1, batch_size=sum_len)
    kron_vec = t3f.get_variable('kron_vec', initializer=kron_vec_init) 

    ans_ = misc._kron_sequence_pairwise_quadratic_form(kron_mat, kron_vec, 
        seq_lens, max_len)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      ans = sess.run(ans_)
      
      # compute answer with numpy
      mat_full = sess.run(t3f.ops.full(kron_mat))
      vec_full = sess.run(t3f.ops.full(kron_vec))
      vec_full = vec_full.reshape([vec_full.shape[0], -1])
      for i, seq_len in enumerate(sess.run(seq_lens)):
        cur_seq = vec_full[:seq_len, :]
        vec_full = vec_full[seq_len:, :]
        ans_np = np.einsum('vi,mij,uj->mvu', cur_seq, mat_full, cur_seq)
        self.assertAllClose(ans_np, ans[:, i, :seq_len, :seq_len], atol=1e-3)

  def test_pairwise_quadratic_form(self):
    """Tests `pairwise_quadratic_form` function.
    """
    pass

  def test_kron_logdet(self):
    """Tests `_kron_logdet` function.
    """
    pass

  def test_kron_tril(self):
    """Tests `_kron_tril` function.
    """
    pass

  def test_mse_r2(self):
    """Tests `mse` and `r2` functions.
    """
    pass

  def test_accuracy_num_correct(self):
    """Tests `accuracy` and `num_correct` function.
    """
    pass

if __name__ == "__main__":
  tf.test.main()
