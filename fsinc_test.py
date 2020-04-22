import numpy as np
import fsinc

import matplotlib.pyplot as plt

def test_1d_u_to_nu():
  x = np.arange(0, 5, .001)
  # x = np.sort(np.random.uniform(0, 5, 5000))
  s = np.sin(2*np.pi*x) + 2 * np.cos(.2 * x)

  xp = np.sort(np.random.uniform(0, 5, 2*x.size))
  print(x)
  print(xp)
  print(x.size, xp.size)
  sp = fsinc.sinc1d_interp_nu3(x, s, xp)

  ssp = np.sin(2*np.pi*xp) + 2 * np.cos(.2 * xp)

  plt.figure()
  plt.plot(x, s, label = 'uniform')
  plt.plot(xp, sp, '--', label = 'interp NU')
  plt.show()

  # np.testing.assert_allclose(sp, ssp)

def test_1d_nu_to_nu():
  x = np.sort(np.random.uniform(-5, 5, 5000))
  s = np.sin(2*np.pi*x) + 2 * np.cos(.2 * x)

  xp = np.sort(np.random.uniform(-5, 5, int(x.size/2)))
  print(x)
  print(xp)
  print(x.size, xp.size)
  sp = fsinc.sinc1d_interp_nu3(x, s, xp)

  ssp = np.sin(2*np.pi*xp) + 2 * np.cos(.2 * xp)

  plt.figure()
  plt.plot(x, s, label = 'uniform')
  plt.plot(xp, sp, '--', label = 'interp NU')
  plt.show()

  # np.testing.assert_allclose(sp, ssp)

