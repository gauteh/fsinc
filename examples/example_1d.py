import numpy as np
import matplotlib.pyplot as plt

import fsinc

## Test uniform to non-uniform sampling
x = np.arange(0, 5, .001)
s = np.sin(2*np.pi*x) + 2 * np.cos(.2 * x)

xp = np.sort(np.random.uniform(.2, 4.8, 10))
sp = fsinc.sinc1d_interp_u(x, s, xp)

ssp = np.sin(2*np.pi*xp) + 2 * np.cos(.2 * xp)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16, 16))
ax1.set_title('Interpolating uniform samples to non-uniform points')
ax1.plot(x, s, label = 'samples')
ax1.plot(xp, sp, 'o', label = 'interpolated values')
ax1.grid()
ax1.legend()

## Test non-uniform to non-uniform sampling
x = np.sort(np.random.uniform(-5, 5, 5000))
s = np.sin(2*np.pi*x) + 2 * np.cos(.2 * x)

xp = np.sort(np.random.uniform(-4, 4, 100))
sp = fsinc.sinc1d_interp_nu2(x, s, xp, 30.)

ax2.set_title('Interpolating non-uniform samples to non-uniform points (Jacobian)')
ax2.plot(x, s, label = 'samples')
ax2.plot(xp, sp, 'o', label = 'interpolated values')
ax2.grid()
ax2.legend()

plt.savefig('example_1d.png')
plt.show()
