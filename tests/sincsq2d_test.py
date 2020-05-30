import numpy as np
import fsinc

def sincsq2d(x, y, s, xp, yp):
  sp = np.zeros((xp.size,))

  for i in range(xp.size):
    for j in range(x.size):
      sp[i] += s[j] * np.sinc(x[j] - xp[i])**2 * np.sinc(y[j] - yp[i])**2

  return sp

def test_sincsq2d_2k(benchmark):
  x = np.linspace(-5, 5, 2000)
  y = np.linspace(-5, 5, 2000)
  s = np.sin(x) * np.cos(y)
  xp = np.linspace(-4.5, 4.4, 2000)
  yp = np.linspace(-4.5, 4.4, 2000)

  benchmark(fsinc.sincsq2d, x, y, s, xp, yp)

def test_sincsq2d_same():
  x = np.linspace(-5, 5, 100)
  y = np.linspace(-5, 5, 100)
  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sincsq2d(x, y, s, x, y, True)
  dp = sincsq2d(x, y, s, x, y)
  np.testing.assert_allclose(sp, dp, atol = 2e-3)

def test_sincsq2d_different_targets():
  x = np.linspace(-5, 5, 100)
  y = np.linspace(-5, 5, 100)

  xp = np.linspace(-4, 3, 50)
  yp = np.linspace(-2, 3.5, 50)

  s = np.random.uniform(-100, 100, x.size)

  sp = fsinc.sincsq2d(x, y, s, xp, yp, True)
  dp = sincsq2d(x, y, s, xp, yp)
  np.testing.assert_allclose(sp, dp, atol = 2e-3)

def test_sincsq2d_nu_to_u():
  x = np.random.uniform(-5, 5, 100)
  y = np.random.uniform(-5, 5, 100)

  xp = np.linspace(-4, 3, 50)
  yp = np.linspace(-2, 3.5, 50)

  s = np.sin(x) * np.cos(y)

  sp = fsinc.sincsq2d(x, y, s, xp, yp, True)
  dp = sincsq2d(x, y, s, xp, yp)
  np.testing.assert_allclose(sp, dp, atol = 2e-3)

def test_sincsq2d_nu_to_nu():
  x = np.random.uniform(-5, 5, 100)
  y = np.random.uniform(-5, 5, 100)

  xp = np.random.uniform(-4, 3, 50)
  yp = np.random.uniform(-2, 3.5, 50)

  s = np.sin(x) * np.cos(y)

  sp = fsinc.sincsq2d(x, y, s, xp, yp, True)
  dp = sincsq2d(x, y, s, xp, yp)
  np.testing.assert_allclose(sp, dp, atol = 2e-3)
