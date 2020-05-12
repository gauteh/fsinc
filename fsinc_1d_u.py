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
