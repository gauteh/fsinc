import numpy as np

from .fsinc_2d import sinc2d

def sinc2d_interp_u(x, y, s, xB, yB, xp, yp):
  """
  Interpolate the uniform samples s(x) onto xp (which could also non-uniform). If
  the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
  reconstructed perfectly.

  Args:
    x (array, floats): uniform sample points
    y (array, floats): uniform sample points
    s (array, floats): uniform sample values at (x, y)
    xB (float): bandwidth, or sampling frequency (1/dx)
    yB (float): bandwidth, or sampling frequency (1/dy)
    xp (array, floats): points of interpolated signal
    yp (array, floats): points of interpolated signal

  Returns:
    sp (array, floats): interpolated signal at (xp, yp).

  """
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert len(s.shape) == 1

  # assert np.max(np.abs(np.diff(x) - (x[1] - x[0]))) < 1.e-15
  # assert np.max(np.abs(np.diff(y) - (y[1] - y[0]))) < 1.e-15

  # xB = 1. / .01 #  np.mean(np.diff(x))
  # yB = 1. / .01 # np.mean(np.diff(y))
  # print('bandwidth:', xB, yB)

  # x = np.arange(0, x.size, 1)
  # y = np.arange(0, y.size, 1)
  x = x * xB
  y = y * yB
  xp = xp * xB
  yp = yp * yB

  return sinc2d(x, y, s, xp, yp, True)

