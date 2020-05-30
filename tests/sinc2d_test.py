import numpy as np
import fsinc

def sinc2d(x, y, s, xp, yp):
  sp = np.zeros((xp.size,))

  for i in range(xp.size):
    for j in range(x.size):
      sp[i] += s[j] * np.sinc(x[j] - xp[i]) * np.sinc(y[j] - yp[i])

  return sp

def test_sinc2d_2k(benchmark):
  x = np.linspace(-5, 5, 2000)
  y = np.linspace(-5, 5, 2000)
  s = np.sin(x)
  xp = np.linspace(-4.5, 4.4, 2000)
  yp = np.linspace(-4.5, 4.4, 2000)

  benchmark(fsinc.sinc2d, x, y, s, xp, yp)

def test_sinc2d_same():
  x = np.linspace(-5, 5, 100)
  y = np.linspace(-5, 5, 100)

  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sinc2d(x, y, s, x, y, True)
  dp = sinc2d(x, y, s, x, y)
  np.testing.assert_almost_equal(sp, dp)

def test_sinc2d_different_targets():
  x = np.linspace(-5, 5, 100)
  y = np.linspace(-5, 5, 100)

  xp = np.linspace(-4, 3, 50)
  yp = np.linspace(-2, 3.5, 50)

  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sinc2d(x, y, s, xp, yp, True)
  dp = sinc2d(x, y, s, xp, yp)
  np.testing.assert_almost_equal(sp, dp)

def test_sinc2d_nu_to_u():
  x = np.random.uniform(-5, 5, 100)
  y = np.random.uniform(-5, 5, 100)

  xp = np.linspace(-4, 3, 50)
  yp = np.linspace(-2, 3.5, 50)

  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sinc2d(x, y, s, xp, yp, True)
  dp = sinc2d(x, y, s, xp, yp)
  np.testing.assert_almost_equal(sp, dp)

def test_sinc2d_nu_to_nu():
  x = np.random.uniform(-5, 5, 100)
  y = np.random.uniform(-5, 5, 100)

  xp = np.random.uniform(-4, 3, 50)
  yp = np.random.uniform(-2, 3.5, 50)

  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sinc2d(x, y, s, xp, yp, True)
  dp = sinc2d(x, y, s, xp, yp)
  np.testing.assert_almost_equal(sp, dp)
