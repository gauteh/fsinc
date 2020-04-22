import numpy as np
import finufftpy as nufft
import fastgl

def sinc1d(x, s, xp):
  """
  Interpolate the non-uniform samples s(x) onto xp which could also be non-uniform. If
  the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
  reconstructed perfectly.

  Args:
    x (array, floats): sample points
    s (array, floats): sample values
    xp (array, floats): points of interpolated signal

  Returns:
    sp (array, floats): interpolated signal at xp.

  """

  assert len(x) == len(s)

  eps = 1.e-15

  # use normalized sinc
  x = x * np.pi
  xp = xp * np.pi
  xm = np.max( [np.max(np.abs(x)), np.max(np.abs(xp)) ])

  resample = 2 # resample rate
  nx = np.ceil(resample * np.round(xm + 3)).astype('int')

  # calculate Legendre-Gauss quadrature weights
  print('calculate legendre-gauss weights', nx)
  xx = np.zeros((nx,))
  ww = np.zeros((nx,))
  for a in range(nx):
    _, ww[a], xx[a] = fastgl.glpair(nx, a + 1)

  h = np.zeros(xx.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(x, s, -1, eps, xx, h, debug = 1, spread_debug = 1)
  assert status == 0

  # weighted signal
  ws = .5 * h * ww

  sp = np.zeros(xp.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(xx, ws, 1, eps, xp, sp, debug = 1, spread_debug = 1)
  assert status == 0

  if np.all(np.isreal(s)):
    return sp.real
  else:
    return sp

