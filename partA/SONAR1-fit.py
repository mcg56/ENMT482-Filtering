import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show, plot
from scipy.optimize import curve_fit
import math

# Load data
filename = 'sonar1-calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
x_raw, z_raw = data.T
x, z = data.T


# Different model equations

def modelLinear(x, m, c):
    return m * x + c

def modelParabola(x, a, b, c):
    return a + b * x + c * x * x

def modelHyperbole(x, a, b):
    return a + b / x

def modelCubic(x, a, b, c, d):
    return a + b * x + c * x * x + d * x**3

def modelExpoential(x, a, b, c):
    return a + np.exp(-1 * b * (x + c))



# Find best fit whilst removing outliers
def iterativeCurveFit(x, z):

    params, cov = curve_fit(modelLinear, x, z)
    zfit = modelLinear(x, *params)
    zerror = z - zfit
    zerror_1 = z_raw - zfit
    #print(sum(abs(zerror))/len(zerror))

    #Calculating variance
    lookUpTable = []
    step = math.floor(len(x)/10)
    for k in range(0, 10):
        print(z_raw[k*step: (k+1)*step])
        lookUpTable.append(np.var(z_raw[k*step: (k+1)*step]))
    print(lookUpTable)

    #np.savetxt('raw.csv', z_raw, delimiter=',')
    initialTolerance = 2
    desiredTolerance = 0.1
    increment = 0.01
    outlierTolerance = initialTolerance
    endCondition = int((initialTolerance-desiredTolerance)/increment)

    for i in range(0, endCondition):
        for j in range(len(x)-1,0,-1):
            if abs(zerror[j]) > outlierTolerance:
                x = np.delete(x, j)
                z = np.delete(z, j)
        outlierTolerance -= increment
        params, cov = curve_fit(modelLinear, x, z)
        zfit = modelLinear(x, *params)
        zerror = z - zfit

    #print(sum(abs(zerror))/len(zerror))
    #plt.plot(x, z, '.')

    #print(params)
    fig, axes = subplots(2,2)
    fig.suptitle('Test Fit')


    axes[0,0].plot(x_raw, z_raw, '.', alpha=0.2)
    axes[0,0].set_title('SONAR 1')
    axes[0,0].set_ylim(bottom=-0.5,top=6)
    axes[0,0].grid()

    axes[1,0].plot(x, z, '.', alpha=0.2)
    axes[1,0].set_title('SONAR 1 Cleaned')
    axes[1,0].set_ylim(bottom=-0.5,top=6)
    axes[1,0].grid()

    axes[1,0].plot(x, zfit)
    axes[0,0].plot(x, zfit)


    axes[0,1].plot(x_raw, zerror_1, '.', alpha=0.2)
    axes[0,1].set_title('SONAR 1 Error')
    #axes[0,1].set_ylim(bottom=-0.5,top=6)
    axes[0,1].grid()

    axes[1,1].plot(x, zerror, '.', alpha=0.2)
    axes[1,1].set_title('SONAR 1 Cleaned Error')
    #axes[1,1].set_ylim(bottom=-0.5,top=6)
    axes[1,1].grid()


    show()

iterativeCurveFit(x, z)


