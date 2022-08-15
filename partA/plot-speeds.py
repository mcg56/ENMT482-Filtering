import numpy as np
from matplotlib.pyplot import subplots, show

class Data_t:
    def __init__(self, filename):
        self.filename = filename
        self.index = []
        self.time = []
        self.distance = []
        self.velocity_command = []
        self.raw_ir1 = []
        self.raw_ir2 = []
        self.raw_ir3 = []
        self.raw_ir4 = []
        self.sonar1 = []
        self.sonar2 = []

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T

class TrainingData_t(Data_t):
    def __init__(self, filename):
        Data_t.__init__(self, filename)
        self.distance = []

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.distance, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T
        self.time_step = self.time[1] - self.time[0]

data = TrainingData_t('training2.csv')
data.load_data()


velocity_estimate = (data.distance[1:] - data.distance[:-1])/(data.time[1:]-data.time[:-1])

velocity_modelled = [0]
for index in range(len(data.velocity_command) - 1):
    new_velocity = data.velocity_command[index]
    if (abs(data.velocity_command[index]) > abs(velocity_modelled[index])):
        new_velocity -= (data.velocity_command[index] - velocity_modelled[index])*0.92
    velocity_modelled.append(new_velocity)

velocity_error = velocity_modelled[1:] - velocity_estimate

distance_modelled = [0]
distance_crude_model = [0]

for index in range(len(data.velocity_command) - 1):
    distance_modelled.append(distance_modelled[index] + velocity_modelled[index]*(data.time[index+1]-data.time[index]))
    distance_crude_model.append(distance_crude_model[index]+data.velocity_command[index]*(data.time[index+1]-data.time[index]))


fig, axes = subplots(4)
fig.suptitle('Motion Model')

axes[0].set_title('Command vs estimate')
axes[0].plot(data.time, data.velocity_command, '-')
axes[0].plot(data.time[1:], velocity_estimate, '-')
axes[0].plot(data.time, velocity_modelled)
axes[0].legend(["command", "actual", "modelled"])

axes[1].plot(data.time, distance_crude_model)
axes[1].plot(data.time, data.distance)
axes[1].plot(data.time, distance_modelled)
axes[1].legend(["command", "actual", "modelled"])

axes[2].plot(data.time[1:], velocity_error, '.')

axes[3].hist(velocity_error, bins=30)

print(np.var(velocity_error))
print(np.mean(velocity_error))

show()
