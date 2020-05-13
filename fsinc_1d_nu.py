import numba
import numpy as np
import finufftpy as nufft
import fastgl


def sinc1d_interp_nu2(x, s, xp, B = 3.):
  """
  Interpolate the non-uniform samples s(x) onto xp (which could also be non-uniform).

  This uses Jacobian weighting (the difference between the samples as weighting).

  Args:
    x (array, floats): non-uniform sample points
    s (array, floats): non-uniform sample values
    xp (array, floats): points of interpolated signal
    B (float): bandlimit of s(x) (default: 3.)

  Returns:
    sp (array, floats): interpolated signal at xp.
  """

  print('mean bandlimit:', 1./np.max(np.diff(x)))

  B = np.float(B)
  print('bandlimit:', B)

  ws = np.diff(x) # use difference as weights
  ws = np.append(ws, ws[-1])
  return (B / np.pi) * sinc1d(B * x, ws * s, B * xp)

def sinc1d_interp_nu3(x, s, xp, B = 3.):
  """
  Interpolate the non-uniform samples s(x) onto xp (which could also be non-uniform).

  This uses a sinc2 weighting of the non-uniform samples according to eq. 34 in Choi and
  Munson, 1998. Or what is referred to as type Sinc-3.

  Args:
    x (array, floats): non-uniform sample points
    s (array, floats): non-uniform sample values
    xp (array, floats): points of interpolated signal
    B (float): bandlimit of s(x) (default: 3.)

  Returns:
    sp (array, floats): interpolated signal at xp.
  """

  print('mean bandlimit:', 1./np.max(np.diff(x)))

  B = np.float(B)
  print('bandlimit:', B)

  ws = (np.pi / B) / sincsq1d(B * x, B * x, np.ones(x.shape)) # use sinc^2 weights
  return (B / np.pi) * sinc1d(B * x, ws * s, B * xp)

def sinc1d(x, s, xp):
  eps = 1.e-15

  # normalized sinc
  # x = x * np.pi
  # xp = xp * np.pi

  xm = np.max( [np.max(np.abs(x)), np.max(np.abs(xp)) ])

  resample = 2 # resample rate
  nx = np.ceil(resample * np.round(xm + 3)).astype('int')

  print('calculate Legendre-Gauss weights (using fastgl)', nx)
  xx, ww = fastgl.lgwt(nx)

  print('nufft1')
  h = np.zeros(xx.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(x, s, -1, eps, xx, h, debug = 1, spread_debug = 1)
  assert status == 0

  # integrated signal
  ws = h * ww

  print('nufft2')
  sp = np.zeros(xp.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(xx, ws, 1, eps, xp, sp, debug = 1, spread_debug = 1)
  sp = .5 * sp
  assert status == 0

  if np.all(np.isreal(s)):
    return sp.real
  else:
    return sp

def sincsq1d(x, s, xp):
  assert len(x) == len(s)

  eps = 1.e-15

  # normalized sinc
  # x = x * np.pi
  # xp = xp * np.pi

  xm = np.max( [np.max(np.abs(x)), np.max(np.abs(xp)) ])

  resample = 2 # resample rate
  nx = np.ceil(resample * np.round(xm + 3)).astype('int')

  # calculate Legendre-Gauss quadrature weights
  print('calculate Legendre-Gauss weights (using fastgl)', nx)
  xx, ww = fastgl.lgwt(nx)
  xx = np.concatenate((xx-1, xx+1)) # covers [-2, 2]
  ww = np.concatenate((ww, ww))

  print('nufft1')
  h = np.zeros(xx.shape, dtype = np.complex128) # signal at xx
  status = nufft.nufft1d3(x, s, -1, eps, xx, h, debug = 1, spread_debug = 1)
  assert status == 0

  # integrated signal
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
