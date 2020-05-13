import numpy as np

from .fsinc_1d import *

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

