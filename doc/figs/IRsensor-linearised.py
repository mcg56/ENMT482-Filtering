import numpy as np
from matplotlib.pyplot import subplots, show, style, savefig
style.use('plot.mplstyle')

x = np.linspace(0, 10, 200)

R_mu = 10
V_sigma = 1

k1 = 2
k2 = 2
z = k1 / (x + k2)

x0 = 5

# Gradient
hp = -k1 / (x0 + k2)**2

# Linearised approximation
za = hp * (x - x0) + k1 / (x0 + k2)

fig, axes = subplots()
axes.plot(x, z)
axes.plot(x, za, '--')
axes.grid()
axes.set_xlabel('Range $x$')
axes.set_ylabel('Measurement $z$')
axes.set_xlim(0, 10)

fig.tight_layout()

savefig(__file__.replace('.py', '.pdf'), bbox_inches='tight')
savefig(__file__.replace('.py', '.pgf'), bbox_inches='tight')
