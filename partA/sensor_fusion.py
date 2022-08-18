import numpy as np
from matplotlib.pyplot import *
from sensor_models import *

class Data_t:
    def __init__(self, filename):
        """Class for reading command and sensor data in order to create Kalman Filter"""
        self.filename = filename
        self.index = []
        self.time = []
        self.velocity_command = []
        self.raw_ir1 = []
        self.raw_ir2 = []
        self.raw_ir3 = []
        self.raw_ir4 = []
        self.sonar1 = []
        self.sonar2 = []
        self.time_step = 0

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T
        self.time_step = self.time[1] - self.time[0]

class TrainingData_t(Data_t):
    """Class that inherits functionality from Data_t but adds distance variable"""
    def __init__(self, filename):
        Data_t.__init__(self, filename)
        self.distance = []

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.distance, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T
        self.time_step = self.time[1] - self.time[0]

class Kalman_filter_t:
    def __init__(self, initial_position, initial_variance):
        self.current_velocity = 0        
        self.position = [initial_position]
        self.variance = [initial_variance]

    def update_prior(self, velocity_command, time_step):
        new_velocity = velocity_command
        if (abs(velocity_command) > abs(self.current_velocity)):
            new_velocity -= (velocity_command - self.current_velocity)*0.92
        self.current_velocity = new_velocity
        self.position.append(self.position[-1] + self.current_velocity*time_step)

    def update_posterior(self):
        pass

class Sensor_t:
    def __init__(self):
        """Class for reading command and sensor data in order to create Kalman Filter"""
        self.parameters = []
        self.model_variance = []
        self.mle_variance = []
        self.mle = []
        self.equations = []

def plot_data(data: TrainingData_t, filter: Kalman_filter_t):
    fig, axes = subplots(2)
    fig.suptitle('Motion Model')

    axes[1].plot(data.time, data.distance)
    axes[1].plot(data.time, filter.position)
    axes[1].legend(["actual", "modelled"])

    show()

def main():
    # data = Data_t('test.csv')
    data = TrainingData_t("training2.csv")
    data.load_data()

    initial_position = 0
    initial_variance = 0

    # Init Sensors
    sonar = Sensor_t()
    ir3 = Sensor_t()
    ir4 = Sensor_t()

    # Sensor Model Estimation
    sonar.parameters, sonar.model_variance = modelSonar()
    ir3.parameters, ir3.model_variance = modelIR3()
    ir4.parameters, ir4.model_variance = modelIR4()

    ######### TESTING ############
    print(len(data.sonar1))
    sonarArray = [0]
    ##############################

    filter = Kalman_filter_t(initial_position, initial_variance)

    for index in range(1, len(data.time), 1):
        filter.update_prior(data.velocity_command[index], data.time_step)
        filter.update_posterior()

        # Inverse sonar model trial
        sonar.mle = inverseLinear(data.sonar1[index], sonar.parameters[0], sonar.parameters[1])
        sonarArray.append(sonar.mle) # TESTING

        # Inverse IR model trial


    ############## TESTING ##############
    plot_data(data, filter)
    fig, axes = subplots(2, 1)
    fig.suptitle('Calibration data')
    axes[0].plot(data.time, sonarArray, '.', alpha=0.2, label="True distance")
    axes[0].plot(data.time, data.distance, '.', alpha=0.2, label="True distance")
    show()
    ####################################

if __name__ == "__main__":
    main()