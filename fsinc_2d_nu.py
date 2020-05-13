import numpy as np

from .fsinc_2d import sinc2d

def sinc2d_interp_nu2(x, y, s, B, xp, yp):
  """
  Interpolate the non-uniform samples s(x) onto xp (which could also non-uniform). If
  the samples s(x) are uniform and satisfy the Nyquist-criterion the signal at `xp` is
  reconstructed perfectly.

  This uses Jacobian weighting, Sinc-2 in Choi & Munson, 1998.

  Args:
    x (array, floats): uniform sample points
    y (array, floats): uniform sample points
    s (array, floats): uniform sample values at (x, y)
    B (float): approx. bandwidth, or sampling frequency (1/dx and 1/dy)
    xp (array, floats): points of interpolated signal
    yp (array, floats): points of interpolated signal

  Returns:
    sp (array, floats): interpolated signal at (xp, yp).

  """
  assert len(x.shape) == 1
  assert len(y.shape) == 1
  assert len(s.shape) == 1

  B = np.float(B)

  wsx = np.diff(x) # use difference as weights
  wsx = np.append(wsx, wsx[-1])

  wsy = np.diff(y)
  wsy = np.append(wsy, wsy[-1])
  wsy.shape = (wsy.shape[0], 1)

  ws = np.dot(wsx, wsy).ravel()

  return (B / np.pi) * sinc2d(B * x, B * y, s, B * xp, B * yp)


