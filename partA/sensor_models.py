import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import curve_fit
import math

class Sensor_t:
    def __init__(self):
        """Class for holding parameters relating to the sensors"""
        self.parameters = 0
        self.model_variance = 0


def modelLinear(x, m, c):
    return m * x + c

def modelParabola(x, a, b, c):
    return a + b * x + c * x * x

def modelHyperbole(x, a, b, c):
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

    # Wariables
    BREAK_POINT_1 = 1000
    # initialTolerance = 2
    # desiredTolerance = 0.1
    # increment = 0.01
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

    # TODO make a module that returns all dis
    params, cov = curve_fit(modelHyperbole, x_1, z_1)
    z_1fit = modelHyperbole(x_1, *params)
    z_error_1 = z_1 - z_1fit

    params, cov = curve_fit(modelLinear, x_2, z_2)
    z_2fit = modelLinear(x_2, *params)
    z_error_2 = z_2 - z_2fit
    
    # zerror = np.concatenate((z_error_1), axis=None)      
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
        fig.suptitle('Test Fit')

        axes[0].plot(distance, raw_ir3, '.', alpha=0.2)
        axes[0].set_title('IR3')

        axes[0].plot(x_1, z_1fit, '.', alpha=0.2, linestyle='-')
        axes[0].set_title('Fit')

        axes[0].plot(x_2, z_2fit, '.', alpha=0.2)
        axes[0].set_title('Fit')

        axes[1].plot(distance, zerror, '.', alpha=0.2)
        axes[1].set_title('Fit')
        show()

    result = [params_1, params_2], [var_1, var_2]
    return result


def modelIR4(plot=False):
    
    # Wariables
    BREAK_POINT_1 = 362
    BREAK_POINT_2 = 655
    # initialTolerance = 2
    # desiredTolerance = 0.1
    # increment = 0.01
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

    # TODO make a module that returns all dis
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
    z_error_3_1 = z_3_1 - z_3_1fit
    var_3 = np.var(z_error_3_1)



    if (plot == True):
        fig, axes = subplots(2)
        fig.suptitle('Test Fit')

        axes[0].plot(distance, raw_ir4, '.', alpha=0.2)
        axes[0].set_title('IR4')

        axes[0].plot(x_1_1, z_1_1fit, '.', alpha=0.2, linestyle='-')


        axes[0].plot(x_2_1, z_2_1fit, '.', alpha=0.2)
        axes[0].plot(x_3_1, z_3_1fit, '.', alpha=0.2)


        axes[1].plot(distance, zerror, '.', alpha=0.2)
        axes[1].set_title('Fit')
        show()

    result = [params_1, params_2, params_3], [var_1, var_2, var_3]
    return result
    
def modelSonar(plot=False):
    # Wariables
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

    # TODO make a module that returns all dis
    params, cov = curve_fit(modelLinear, x_1, z_1)
    z_1fit = modelLinear(x_1, *params)
    z_error_1 = z_1 - z_1fit
    var_1 = np.var(z_error_1[:100])
    # print(var_1) #TESTING



    # Remove outliers and remodel
    x_1_1,z_1_1 = iterative_fitting(modelLinear, final_deviation, x_1, z_1, z_error_1)
    params_1, cov = curve_fit(modelLinear, x_1_1, z_1_1)
    z_1_1fit = modelLinear(x_1_1, *params_1)
    z_error = z_1_1 - z_1_1fit

    # # Find varience of new fit. Omit outliers
    # var_lut = []
    # bin_width = math.floor(len(x_1_1)/10)
    # bin_step = x_1_1[-1]/10

    # for bin in range(10):
    #     LUT_range = (bin*bin_step, (bin+1)*bin_step)
    #     var = np.var(z_error[bin*bin_width: (bin+1)*bin_width])
    #     var_lut.append([LUT_range, var])
    # print(var_lut)

    var = np.var(z_error)
    # print(np.var(z_error))

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

sonar = Sensor_t()
ir3 = Sensor_t()
ir4 = Sensor_t()

# Sensor Model Estimation
sonar.parameters, sonar.model_variance = modelSonar(True)
print(sonar.parameters)
print(sonar.model_variance)
# ir3.parameters, ir3.model_variance = modelIR3()
# ir4.parameters, ir4.model_variance = modelIR4()

# # Inverse sonar model trial
# sonar.mle = inverseLinear(data.sonar1[index], sonar.parameters[0], sonar.parameters[1])
# sonarArray.append(sonar.mle) # TESTING