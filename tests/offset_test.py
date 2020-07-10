import numpy as np
from fsinc import zero_offset

def test_offset():
  x = np.array([-5, -4, 1, 10])
  xp = np.array([0, 9])
  x, xp = zero_offset(x, xp)
  np.testing.assert_equal(x, [0, 1, 6, 15])
  np.testing.assert_equal(xp, [5, 14])
