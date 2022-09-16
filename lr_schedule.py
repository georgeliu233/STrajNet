import tensorflow as tf
import math 

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "SGDRDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      first_decay_steps = tf.cast(self.first_decay_steps, dtype)
      alpha = tf.cast(self.alpha, dtype)
      t_mul = tf.cast(self._t_mul, dtype)
      m_mul = tf.cast(self._m_mul, dtype)

      global_step_recomp = tf.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        """Helper for `cond` operation."""
        if geometric:
          i_restart = tf.floor(
              tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              tf.math.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = tf.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = tf.cond(
          tf.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
          tf.constant(math.pi, dtype=dtype) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return tf.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }

import matplotlib.pyplot as plt

def test_img():
  # sch = CustomSchedule(384, warmup_steps=60000)
  sch = CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=30000,
    t_mul=1.2,
    m_mul=1.0,
    alpha=5e-6,
    name=None)
  plt.figure()
  plt.plot(sch(tf.range(30000*16,dtype=tf.float32)))


if __name__=='__main__':
  test_img()