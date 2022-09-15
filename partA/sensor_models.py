"""Module to find sensor models.

S.W. Bain and M.C. Gardyne,
15/08/2022
"""

import numpy as np
from matplotlib.pyplot import *
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit

class Sensor_t:
    def __init__(self):
        """Class for holding parameters relating to the sensors"""
        self.parameters = 0
        self.model_variance = 0


def modelLinear(x, a, b):
    return a + b*x

def modelParabola(x, a, b, c):
    return a + b * x + c * x * x

def inverseParabola(z, a, b, c):
    return np.roots(c, b, a - z)

def modelHyperbole(x, a, b):
    return a + b / x 

def modelHyperbole_1(x, a, b, c):
    return a + b / x + c*x

def inverseLinear(z, m, c):
    return (z - c)/m

def inverseHyperbola(z, a, b):
    return a/(z-b)

def iterative_fitting(curve, final_deviation, x, z, z_error):
    deviation = np.std(z_error)
    while (deviation > final_deviation):
        for i in range(len(x)-1,0,-1):
            if abs(z_error[i]) > deviation:
                x = np.delete(x, i)
                z = np.delete(z, i)

        params, cov = curve_fit(curve, x, z)
        zfit = curve(x, *params)
        z_error = z - zfit
        deviation = np.std(z_error)
    
    return x, z


def modelIR3(plot=False):

    BREAK_POINT_1 = 1000
    final_deviation = 0.1

    # Load data
    filename = 'calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
        sonar1, sonar2 = data.T

    # Non Linear Region
    x_1 = distance[0:BREAK_POINT_1]
    z_1 = raw_ir3[0:BREAK_POINT_1]

    # # Linear Region
    x_2 = distance[BREAK_POINT_1:]
    z_2 = raw_ir3[BREAK_POINT_1:]

    params, cov = curve_fit(modelHyperbole, x_1, z_1)
    z_1fit = modelHyperbole(x_1, *params)
    z_error_1 = z_1 - z_1fit

    params, cov = curve_fit(modelLinear, x_2, z_2)
    z_2fit = modelLinear(x_2, *params)
    z_error_2 = z_2 - z_2fit
          
    zerror = np.concatenate((z_error_1, z_error_2), axis=None)

    # Remove outliers and remodel
    x_1_1,z_1_1 = iterative_fitting(modelHyperbole, final_deviation, x_1, z_1, z_error_1)
    params_1, cov = curve_fit(modelHyperbole, x_1_1, z_1_1)
    z_1_1fit = modelHyperbole(x_1_1, *params_1)
    z_error_1_1 = z_1_1 - z_1_1fit
    var_1 = np.var(z_error_1_1)

    # Remove outliers and remodel
    x_2_1,z_2_1 = iterative_fitting(modelLinear, final_deviation, x_2, z_2, z_error_2)
    params_2, cov = curve_fit(modelLinear, x_2_1, z_2_1)
    z_2_1fit = modelLinear(x_2_1, *params_2)
    z_error_2_1 = z_2_1 - z_2_1fit
    var_2 = np.var(z_error_2_1)


    if (plot == True):
        fig, axes = subplots(2)

        axes[0].plot(distance, raw_ir3, '.', alpha=0.2)
        axes[0].plot(x_1, z_1fit, '.', alpha=0.2, linestyle='-')
        axes[0].plot(x_2, z_2fit, '.', alpha=0.2)

        axes[1].plot(distance, zerror, '.', alpha=0.2)

        show()

    result = [params_1, params_2], [var_1, var_2]
    return result


def modelIR4(plot=False):
    
    BREAK_POINT_1 = 362
    BREAK_POINT_2 = 655
    final_deviation = 0.1

    # Load data
    filename = 'calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
        sonar1, sonar2 = data.T

    # Parabolic Region
    x_1 = distance[0:BREAK_POINT_1]
    z_1 = raw_ir4[0:BREAK_POINT_1]

    # Linear Region
    x_2 = distance[BREAK_POINT_1:BREAK_POINT_2]
    z_2 = raw_ir4[BREAK_POINT_1:BREAK_POINT_2]

    # Hyperbolic region
    x_3 = distance[BREAK_POINT_2:]
    z_3 = raw_ir4[BREAK_POINT_2:]


    params, cov = curve_fit(modelParabola, x_1, z_1)
    z_1fit = modelParabola(x_1, *params)
    z_error_1 = z_1 - z_1fit

    params, cov = curve_fit(modelLinear, x_2, z_2)
    z_2fit = modelLinear(x_2, *params)
    z_error_2 = z_2 - z_2fit

    params, cov = curve_fit(modelHyperbole, x_3, z_3)
    z_3fit = modelHyperbole(x_3, *params)
    z_error_3 = z_3 - z_3fit

    zerror = np.concatenate((z_error_1, z_error_2, z_error_3), axis=None)

    # Remove outliers and remodel
    x_1_1,z_1_1 = iterative_fitting(modelParabola, final_deviation, x_1, z_1, z_error_1)
    params_1, cov = curve_fit(modelParabola, x_1_1, z_1_1)
    z_1_1fit = modelParabola(x_1_1, *params_1)
    z_error_1_1 = z_1_1 - z_1_1fit
    var_1 = np.var(z_error_1_1)

    # Remove outliers and remodel
    x_2_1,z_2_1 = iterative_fitting(modelLinear, final_deviation, x_2, z_2, z_error_2)
    params_2, cov = curve_fit(modelLinear, x_2_1, z_2_1)
    z_2_1fit = modelLinear(x_2_1, *params_2)
    z_error_2_1 = z_2_1 - z_2_1fit
    var_2 = np.var(z_error_2_1)

    # Remove outliers and remodel
    x_3_1,z_3_1 = iterative_fitting(modelHyperbole, final_deviation, x_3, z_3, z_error_3)
    params_3, cov = curve_fit(modelHyperbole, x_3_1, z_3_1)
    z_3_1fit = modelHyperbole(x_3_1, *params_3)
    fitted_data = np.multiply(1/distance[BREAK_POINT_2:],1.4931097) + 1.25294805
    z_error_3_1 = fitted_data - raw_ir4[BREAK_POINT_2:]
    var_3 = np.var(z_error_3_1)



    if (plot == True):
        fig, axes = subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})

        axes[0].plot(distance, raw_ir4, '.', alpha=0.2)
        axes[0].set_ylabel('z(x)')
        axes[0].set_xlabel('x [m]')
        axes[0].grid()
        axes[0].plot(x_3_1, z_3_1fit, '.', alpha=0.2)

        axes[1].hist(z_error_3_1, bins= 30, weights=np.ones(len(z_error_3_1))/len(z_error_3_1))
        axes[1].set_ylabel('Probabiltiy desnity')
        axes[1].set_xlabel('Error [m]')
        axes[1].grid()
        gca().yaxis.set_major_formatter(PercentFormatter(1))

        show()

    result = [params_1, params_2, params_3], [var_1, var_2, var_3]
    return result
    
def modelSonar(plot=False):

    final_deviation = 0.1

    # Load data
    filename = 'calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
        sonar1, sonar2 = data.T

    # Linear Region
    x_1 = distance
    z_1 = sonar1

    params, cov = curve_fit(modelLinear, x_1, z_1)
    z_1fit = modelLinear(x_1, *params)
    z_error_1 = z_1 - z_1fit
    var_1 = np.var(z_error_1[:100])

    # Remove outliers and remodel
    x_1_1,z_1_1 = iterative_fitting(modelLinear, final_deviation, x_1, z_1, z_error_1)
    params_1, cov = curve_fit(modelLinear, x_1_1, z_1_1)
    z_1_1fit = modelLinear(x_1_1, *params_1)
    z_error = z_1_1 - z_1_1fit

    var = np.var(z_error)

    if (plot == True):
        fig, axes = subplots(2)
        fig.suptitle('Test Fit')

        axes[0].plot(x_1, sonar1, '.', alpha=0.2)
        axes[0].set_title('Sonar 1')
        axes[0].plot(x_1_1, z_1_1fit, '.', alpha=0.2, linestyle='-')
        axes[0].plot(x_1, z_1fit, '.', alpha=0.2, linestyle='-')

        axes[1].plot(x_1_1, z_1_1, '.', alpha=0.2)
        axes[1].set_title('Fit')

        show()
    return params_1, var


# modelIR3(True)
# modelIR4(True)
# modelSonar(True)

