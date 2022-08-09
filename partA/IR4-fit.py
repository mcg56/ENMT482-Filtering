import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit
import math


# Load data
filename = 'partA/calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T


def modelLinear(x, m, c):
    return m * x + c

def modelParabola(x, a, b, c):
    return a + b * x + c * x * x

def modelHyperbole(x, a, b):
    return a + b / x

def modelCubic(x, a, b, c, d):
    return a + b * x + c * x * x + d * x**3

def modelPoly(x, a, b, c, d, e, f, g):
    return a + b * x + c * x * x + d * x**3

def modelExpoential(x, a, b, c):
    return a + np.exp(-1 * b * (x + c))

BREAK1 = 380
BREAK2 = 600



# Non Linear
x_1 = distance[BREAK2:]
z_1 = raw_ir4[BREAK2:]

# Linear 1
x_2 = distance[:BREAK1]
z_2 = raw_ir4[:BREAK1]

# Linear 2
x_3 = distance[BREAK1:BREAK2]
z_3 = raw_ir4[BREAK1:BREAK2]

params, cov = curve_fit(modelHyperbole, x_1, z_1)
z_1fit = modelHyperbole(distance, *params)

params, cov = curve_fit(modelParabola, x_2, z_2)
z_2fit = modelParabola(distance, *params)

params, cov = curve_fit(modelLinear, x_3, z_3)
z_3fit = modelLinear(distance, *params)

zerror_1 = z_1 - z_1fit[BREAK2:]
# zerror_2 = z_2 - z_2fit[END:]

# zerror = np.concatenate((zerror_1,zerror_2), axis=None)

# print(len(zfit))

# print(sum(zerror)/len(zerror))


fig, axes = subplots(2)
fig.suptitle('Test Fit')

axes[0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[0].set_title('IR4')

axes[0].plot(distance, z_1fit, '.', alpha=0.2)

axes[0].plot(distance, z_2fit, '.', alpha=0.2)

axes[0].plot(distance, z_3fit, '.', alpha=0.2)

# axes[1].plot(distance, zerror, '.', alpha=0.2)
# axes[1].set_title('Fit')

axes[0].set_ylim(bottom=0,top=5)

show()



