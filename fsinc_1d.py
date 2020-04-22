import numba
import numpy as np
import finufftpy as nufft
import fastgl


def sinc1d_interp(x, s, xp):
  """
  Interpolate the uniform samples s(x) onto xp (which could also non-uniform). If
  the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
  reconstructed perfectly.

  Args:
    x (array, floats): uniform sample points
    s (array, floats): uniform sample values
    xp (array, floats): points of interpolated signal

  Returns:
    sp (array, floats): interpolated signal at xp.

  """
  assert np.max(np.abs(np.diff(x) - (x[1] - x[0]))) < 1.e-15

  B = 1. / np.mean(np.diff(x))
  print('bandwidth:', B)

  x = np.arange(0, x.size, 1)
  xp = xp * B

  return sinc1d(x, s, xp)

def sinc1d_interp_nu3(x, s, xp):
  """
  Interpolate the non-uniform samples s(x) onto xp (which could also be non-uniform).

  This uses a sinc2 weighting of the non-uniform samples according to eq. 34 in Choi and
  Munson, 1998.

  Args:
    x (array, floats): non-uniform sample points
    s (array, floats): non-uniform sample values
    xp (array, floats): points of interpolated signal

  Returns:
    sp (array, floats): interpolated signal at xp.

  """
  B = 1. / np.mean(np.diff(x))
  # B = 3.3
  # B = 4.
  # B = 10
  print('bandwidth:', B)

  # x = np.arange(0, x.size, 1)
  # xp = xp * B

  ws = (np.pi / B) / sincsq1d(B * x, B * x, np.ones(x.shape))
  return (B / np.pi) * sinc1d(B * x, ws * s, B * xp)

def sinc1d(x, s, xp):
  """
  Expects x to be normalized from 0 to N integers, and xp to be normalized with
  bandwidth of x.
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
  print('calculate Legendre-Gauss weights (using fastgl)', nx)
  xx, ww = fastgl.lgwt(nx)

  print('nufft1')
  h = np.zeros(xx.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(x, s, -1, eps, xx, h, debug = 1, spread_debug = 1)
  assert status == 0

  # weighted signal
  ws = .5 * h * ww

  print('nufft2')
  sp = np.zeros(xp.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(xx, ws, 1, eps, xp, sp, debug = 1, spread_debug = 1)
  assert status == 0

  if np.all(np.isreal(s)):
    return sp.real
  else:
    return sp

def sincsq1d(x, s, xp):
  """
  Expects x to be normalized from 0 to N integers, and xp to be normalized with
  bandwidth of x.
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
  print('calculate Legendre-Gauss weights (using fastgl)', nx)
  xx, ww = fastgl.lgwt(nx)
  xx = np.concatenate((xx-1, xx+1))
  ww = np.concatenate((ww, ww))

  print('nufft1')
  h = np.zeros(xx.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(x, s, -1, eps, xx, h, debug = 1, spread_debug = 1)
  assert status == 0

  # weighted signal
  ws = h * ww * (2 - np.abs(xx))

  print('nufft2')
  sp = np.zeros(xp.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(xx, ws, 1, eps, xp, sp, debug = 1, spread_debug = 1)
  sp = sp * 0.25
  assert status == 0

  if np.all(np.isreal(s)):
    return sp.real
  else:
    return sp
