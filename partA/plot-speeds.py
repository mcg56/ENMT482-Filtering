import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T


velocity_estimate = (distance[1:] - distance[:-1])/(time[1:]-time[:-1])
acceleration_estimate = (velocity_estimate[1:] - velocity_estimate[:-1])/(time[2:]-time[:-2])

distance_estimate = [0]
for index in range(len(velocity_command) - 1):
    distance_estimate.append(distance_estimate[index] + velocity_command[index]*time[index]/2000)

fig, axes = subplots(6)
fig.suptitle('Motion Model')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

show()


axes[2].plot(velocity_estimate, velocity_estimate - velocity_command[1:], '.', alpha=0.2)
axes[2].set_title('Estimate error over velocity')

axes[3].plot(acceleration_estimate, velocity_estimate[1:] - velocity_command[2:], '.', alpha=0.2)
axes[3].set_title('Estimate error over acceleration')

axes[4].plot(time[1:], velocity_estimate - velocity_command[1:], '.', alpha=0.2)
axes[4].set_title('Estimate error over time')

axes[5].plot(time, distance, '.', alpha=0.2)
axes[5].plot(time, distance_estimate, '.', alpha=0.2)
axes[5].set_title('Distance over time')

show()
