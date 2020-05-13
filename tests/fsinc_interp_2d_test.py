import numpy as np
import fsinc

import matplotlib.pyplot as plt

def test_2d_u_to_nu():
  x = np.arange(0, 5, .01)
  y = np.arange(1, 4, .01)

  x, y = np.meshgrid(x, y)
  s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)

  # plt.pcolormesh(x, y, s)
  # plt.colorbar()

  x, y, s = x.ravel(), y.ravel(), s.ravel()
  print(x.shape)

  xp = np.sort(np.random.uniform(.2, 4.5, 1000))
  yp = np.sort(np.random.uniform(1.1, 3.5, 1000))
  ps = (xp.size, yp.size)
  xp, yp = np.meshgrid(xp, yp)
  xp, yp = xp.ravel(), yp.ravel()

  sp = fsinc.sinc2d_interp_u(x, y, s, 100, 100, xp, yp)
  xp = xp.reshape(ps)
  yp = yp.reshape(ps)
  sp = sp.reshape(ps)

#   plt.figure()
#   plt.pcolormesh(xp, yp, sp)
#   plt.colorbar()
#   plt.show()

  ssp = np.sin(2*np.pi*xp) * 2 * np.cos(yp*np.pi)

  np.testing.assert_allclose(sp, ssp, rtol = 1.e-1, atol = 2e-2)


def test_2d_nu_to_u_2():
  x = np.sort(np.random.uniform(0, 5, 5000))
  y = np.sort(np.random.uniform(0, 5, 5000))


  wsx = np.diff(x) # use difference as weights
  wsx = np.append(wsx, wsx[-1])

  wsy = np.diff(y)
  wsy = np.append(wsy, wsy[-1])
  wsy.shape = (wsy.shape[0], 1)

  ws = wsx * wsy
  print("ws:", ws.shape)
  ws = ws.ravel()

  x, y = np.meshgrid(x, y)
  pps = x.shape
  x, y = x.ravel(), y.ravel()
  s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)

  plt.pcolormesh(x.reshape(pps), y.reshape(pps), s.reshape(pps))
  plt.colorbar()


  xp = np.arange(.2, 4.5, .1)
  yp = np.arange(.2, 4.5, .1)
  ps = (xp.size, yp.size)
  xp, yp = np.meshgrid(xp, yp)
  xp, yp = xp.ravel(), yp.ravel()

  sp = fsinc.sinc2d_interp_nu2(x, y, s * ws, 100., xp, yp)
  xp = xp.reshape(ps)
  yp = yp.reshape(ps)
  sp = sp.reshape(ps)

  plt.figure()
  plt.pcolormesh(xp, yp, sp)
  plt.colorbar()
  plt.show()

  ssp = np.sin(2*np.pi*xp) * 2 * np.cos(yp*np.pi)

  np.testing.assert_allclose(sp, ssp, rtol = 1.e-1, atol = 2e-2)


