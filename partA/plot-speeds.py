from posixpath import abspath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

# Load data
filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T


velocity_estimate = (distance[1:] - distance[:-1])/(time[1:]-time[:-1])
acceleration_estimate = (velocity_estimate[1:] - velocity_estimate[:-1])/(time[2:]-time[:-2])

velocity_modelled = [0]
for index in range(len(velocity_command) - 1):
    new_velocity = velocity_command[index]
    if (abs(velocity_command[index]) > abs(velocity_modelled[index])):
        new_velocity -= (velocity_command[index] - velocity_modelled[index])*0.92
    velocity_modelled.append(new_velocity)

distance_modelled = [0]
distance_crude_model = [0]

for index in range(len(velocity_command) - 1):
    distance_modelled.append(distance_modelled[index] + velocity_modelled[index]*(time[index+1]-time[index]))
    distance_crude_model.append(distance_crude_model[index]+velocity_command[index]*(time[index+1]-time[index]))


fig, axes = subplots(2)
fig.suptitle('Motion Model')

axes[0].set_title('Command vs estimate')
axes[0].plot(time, velocity_command, '-')
axes[0].plot(time[1:], velocity_estimate, '-')
axes[0].plot(time, velocity_modelled)
axes[0].legend(["command", "actual", "modelled"])

axes[1].plot(time, distance_crude_model)
axes[1].plot(time, distance)
axes[1].plot(time, distance_modelled)
axes[1].legend(["command", "actual", "modelled"])




# axes[1].plot(time[1:], velocity_estimate - velocity_command[1:], '.', alpha=0.2)
# axes[1].set_title('Estimate error over distance')

# axes[2].plot(velocity_estimate, velocity_estimate - velocity_command[1:], '.', alpha=0.2)
# axes[2].set_title('Estimate error over velocity')

# axes[3].plot(acceleration_estimate, velocity_estimate[1:] - velocity_command[2:], '.', alpha=0.2)
# axes[3].set_title('Estimate error over acceleration')

# axes[4].plot(time[1:], velocity_estimate - velocity_command[1:], '.', alpha=0.2)
# axes[4].set_title('Estimate error over time')

# axes[5].plot(time, distance, '.', alpha=0.2)
# axes[5].plot(time, distance_estimate, '.', alpha=0.2)
# axes[5].set_title('Distance over time')

show()
