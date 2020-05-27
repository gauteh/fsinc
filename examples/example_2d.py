import numpy as np
import matplotlib.pyplot as plt

import fsinc

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16, 16))

## Interpolate uniform points to non-uniform
x = np.arange(0, 5, .01)
y = np.arange(1, 4, .01)

x, y = np.meshgrid(x, y)
s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)

ax1.pcolormesh(x, y, s)
ax1.set_title('uniform points')
# ax1.colorbar()

x, y, s = x.ravel(), y.ravel(), s.ravel()

xp = np.sort(np.random.uniform(.2, 4.5, 1000))
yp = np.sort(np.random.uniform(1.1, 3.5, 1000))
ps = (xp.size, yp.size)
xp, yp = np.meshgrid(xp, yp)
xp, yp = xp.ravel(), yp.ravel()

sp = fsinc.sinc2d_interp_u(x, y, s, 100, 100, xp, yp)
xp = xp.reshape(ps)
yp = yp.reshape(ps)
sp = sp.reshape(ps)

ax2.pcolormesh(xp, yp, sp)
ax2.set_title('resampled to non-uniform points')
# ax2.colorbar()

## Interpolate non-uniform points to uniform grid
x = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))
y = np.sort(np.random.uniform(0, 5, np.sqrt(4.e6).astype('int')))

x, y = np.meshgrid(x, y)
pps = x.shape
x, y = x.ravel(), y.ravel()
s = np.sin(2*np.pi*x) * 2 * np.cos(y*np.pi)
print("max,min=", np.max(s), np.min(s))

ax3.pcolormesh(x.reshape(pps), y.reshape(pps), s.reshape(pps))
ax3.set_xlim([.2, 4.5])
ax3.set_ylim([.2, 4.5])
ax3.set_title('non-uniform points')
# ax3.colorbar()

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

m = ax4.pcolormesh(xp, yp, sp)
ax4.set_title('resampled from non-uniform to uniform points')
# fig.colorbar(m, ax4)

plt.savefig('example_2d.png')
plt.show()
