import numpy as np
import fsinc

import matplotlib.pyplot as plt

def test_2d_u_to_nu(plot):
  x = np.arange(0, 5, .01)
  y = np.arange(1, 4, .01)

  x, y = np.meshgrid(x, y)
  s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)

  if plot:
    plt.pcolormesh(x, y, s)
    plt.colorbar()

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

  if plot:
    plt.figure()
    plt.pcolormesh(xp, yp, sp)
    plt.colorbar()
    plt.show()

  ssp = np.sin(2*np.pi*xp) * 2 * np.cos(yp*np.pi)

  np.testing.assert_allclose(sp, ssp, rtol = 1.e-1, atol = 2e-2)


def test_2d_nu_to_u_2(plot):
  x = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))
  y = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))

  x, y = np.meshgrid(x, y)
  pps = x.shape
  x, y = x.ravel(), y.ravel()
  s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)
  print("max,min=", np.max(s), np.min(s))

  if plot:
    plt.pcolormesh(x.reshape(pps), y.reshape(pps), s.reshape(pps))
    plt.xlim([.2, 4.5])
    plt.ylim([.2, 4.5])
    plt.colorbar()


  xp = np.arange(.2, 4.5, .01)
  yp = np.arange(.2, 4.5, .01)
  ps = (xp.size, yp.size)
  xp, yp = np.meshgrid(xp, yp)
  xp, yp = xp.ravel(), yp.ravel()

  sp = fsinc.sinc2d_interp_nu2(x, y, s, 20., xp, yp)
  print("sp -> max,min=", np.max(sp), np.min(sp))
  xp = xp.reshape(ps)
  yp = yp.reshape(ps)
  sp = sp.reshape(ps)

  if plot:
    plt.figure()
    plt.pcolormesh(xp, yp, sp)
    plt.colorbar()
    plt.show()

  ssp = np.sin(2*np.pi*xp) * 2 * np.cos(yp*np.pi)

  # np.testing.assert_allclose(sp, ssp, rtol = 1.e-1, atol = 2e-2)

def test_2d_nu_to_u_3(plot):
  x = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))
  y = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))

  x, y = np.meshgrid(x, y)
  pps = x.shape
  x, y = x.ravel(), y.ravel()
  s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)
  print("max,min=", np.max(s), np.min(s))

  if plot:
    plt.pcolormesh(x.reshape(pps), y.reshape(pps), s.reshape(pps))
    plt.xlim([.2, 4.5])
    plt.ylim([.2, 4.5])
    plt.colorbar()

  xp = np.arange(.2, 4.5, .01)
  yp = np.arange(.2, 4.5, .01)
  ps = (xp.size, yp.size)
  xp, yp = np.meshgrid(xp, yp)
  xp, yp = xp.ravel(), yp.ravel()

  sp = fsinc.sinc2d_interp_nu3(x, y, s, 40., xp, yp)
  print("sp -> max,min=", np.max(sp), np.min(sp))
  xp = xp.reshape(ps)
  yp = yp.reshape(ps)
  sp = sp.reshape(ps)

  if plot:
    plt.figure()
    plt.pcolormesh(xp, yp, sp)
    plt.colorbar()
    plt.show()

  ssp = np.sin(2*np.pi*xp) * 2 * np.cos(yp*np.pi)

  np.testing.assert_allclose(sp, ssp, rtol = 1.e-1, atol = .35)

def test_jacobi_1d():
  x = np.array([1, 2, 3, 4, 8, 12, 16])
  print(fsinc.jacobi_1d(x))

  print(np.gradient(x))
  print(np.gradient(x, np.arange(0, x.size)))
  print(1./np.gradient(np.arange(0, x.size), x))

def test_jacobi_2d():
  x = np.array([1, 2, 3, 4, 8, 12, 16])
  y = np.array([4, 5, 6, 10, 14, 18, 22])

  xx, yy = np.meshgrid(x, y)
  print("xx=", xx)
  print("yy=", yy)

  xx, yy = xx.ravel(), yy.ravel()
  print("xx=", xx)
  print("yy=", yy)

  print("dxx =", fsinc.jacobi_1d(xx))
  print("dyy =", fsinc.jacobi_1d(yy))

