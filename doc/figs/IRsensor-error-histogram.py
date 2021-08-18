import numpy as np
from matplotlib.pyplot import subplots, show, style, savefig
style.use('plot.mplstyle')

x = np.linspace(0, 10, 2000)

R_mu = 10
V_sigma = 1

k1 = 2
k2 = 2
h = k1 / (x + k2)
z = h + np.random.standard_normal(x.shape) * 0.05

v = z - h
v = v[x < 1.0]

fig, axes = subplots()
axes.hist(v, bins=50, density=True)
axes.grid()
axes.set_xlabel('Error $v$')
axes.set_ylabel('Histogram')

fig.tight_layout()

savefig(__file__.replace('.py', '.pdf'), bbox_inches='tight')
savefig(__file__.replace('.py', '.pgf'), bbox_inches='tight')
