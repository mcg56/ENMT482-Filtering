import numpy as np
from matplotlib.pyplot import *
import math

ALPHA = 0.035

class Measurement_t:
    def __init__(self, mle, variance):
        self.mle = mle
        self.variance = variance

class MotionModel_t:
    def __init__(self):
        self.variance = 0.001178525902611975 * 5

class SonarModel_t:
    def __init__(self):
        self.parameters = [0.99343, -0.017464]
        self.model_variance = 0.00053022

    def calcDist(self, sensor_measurement):
        estimate = (sensor_measurement - self.parameters[1])/self.parameters[0]
        variance = self.model_variance/(self.parameters[0]**2)
        return Measurement_t(estimate, variance)

class SensorModels_t:
    def __init__(self):
        self.sonar1_model = SonarModel_t()

    def fuse_sensors(self, sonar1_meas):
        self.sonar1_model.calcDist(sonar1_meas)

class KalmanFilter_t:
    def __init__(self):
        self.current_velocity = 0
        self.motion_model = MotionModel_t()

    def update_prior(self, measurement: Measurement_t, velocity_command, time_step):
        if (abs(velocity_command) > abs(self.current_velocity)):
            self.current_velocity = velocity_command * ALPHA + self.current_velocity * (1 - ALPHA)
        else:
            self.current_velocity = velocity_command

        position_estimate = measurement.mle + self.current_velocity*time_step
        position_variance = measurement.variance + self.motion_model.variance
        return Measurement_t(position_estimate, position_variance)


    def update_posterior(self, prior: Measurement_t, sensor_measurement: Measurement_t):
        
        if abs(sensor_measurement.mle - prior.mle) < math.sqrt(sensor_measurement.variance) + math.sqrt(prior.variance):
            kalman_gain = (1/sensor_measurement.variance)/(1/prior.variance + 1/sensor_measurement.variance)
            position_estimate = kalman_gain * sensor_measurement.mle + (1-kalman_gain) * prior.mle
            position_variance = 1/(1/prior.variance + 1/sensor_measurement.variance)
            posterior = Measurement_t(position_estimate, position_variance)
        else:
            posterior = prior

        return posterior

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

        self.sensors_models = SensorModels_t()

        self.measurements = []
        self.priors = []
        self.just_motion = []
        self.just_sensor = []

        self.filter = KalmanFilter_t()

    def time_step(self, index):
        return (self.time[index]-self.time[index-1])

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T

    def run_data(self):
        self.measurements.append(self.sensors_models.sonar1_model.calcDist(self.sonar1[0]))

        self.just_sensor.append(self.measurements[0])
        self.just_motion.append(self.measurements[0])

        for index in range(1, len(self.time)):
            self.priors.append(self.filter.update_prior(self.measurements[-1], self.velocity_command[index], self.time_step(index)))
            self.measurements.append(self.filter.update_posterior(self.priors[-1], self.sensors_models.sonar1_model.calcDist(self.sonar1[index])))
            
            self.just_motion.append(self.filter.update_prior(self.just_motion[-1], self.velocity_command[index], self.time_step(index)))
            self.just_sensor.append(self.sensors_models.sonar1_model.calcDist(self.sonar1[index]))

    def plot_data(self):
        fig, axes = subplots(2)
        fig.suptitle('Motion Model')

        axes.plot(self.time, [measurement.mle for measurement in self.measurements])

        show()

class TrainingData_t(Data_t):
    """Class that inherits functionality from Data_t but adds distance variable"""
    def __init__(self, filename):
        Data_t.__init__(self, filename)
        self.distance = []

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.distance, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T

    def plot_data(self):
        fig, axes = subplots(2)
        fig.suptitle('Kalman Filter')

        axes[0].plot(self.time, self.distance)
        axes[0].plot(self.time, [measurement.mle for measurement in self.just_sensor])
        axes[0].plot(self.time, [measurement.mle for measurement in self.measurements])
        axes[0].plot(self.time, [measurement.mle for measurement in self.just_motion])
        axes[0].legend(['Actual', 'Just Sensor', 'Predicted', 'Just Motion'])

        axes[1].plot(self.time, [measurement.variance for measurement in self.measurements])

        show()




def main():
    # data = Data_t('test.csv')
    data = TrainingData_t("training2.csv")
    data.load_data()
    data.run_data()
    data.plot_data()

if __name__ == "__main__":
    main()