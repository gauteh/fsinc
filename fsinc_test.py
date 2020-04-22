import numpy as np
import fsinc

import matplotlib.pyplot as plt

def test_1d_u_to_nu():
  x = np.arange(0, 5, .001)
  s = np.sin(2*np.pi*x)

  xp = np.sort(np.random.uniform(0, 5, 2*x.size))
  print(x)
  print(xp)
  print(x.size, xp.size)
  sp = fsinc.sinc1d(np.arange(0, 5000, 1), s, xp/.001)

  ssp = np.sin(2*np.pi*xp)

  plt.figure()
  plt.plot(x, s, label = 'uniform')
  plt.plot(xp, sp, '--', label = 'interp NU')
  plt.show()

  np.testing.assert_allclose(sp, ssp)

