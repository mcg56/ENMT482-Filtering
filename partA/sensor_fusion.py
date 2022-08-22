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
        self.velocity_variance = 0.001178525902611975

class SensorModel_t:
    def __init__(self, parameters, variance, range, fit_type):
        self.parameters = parameters
        self.model_variance = variance
        self.range = range
        self.fit_type = fit_type

        self.raw_data = []

    def check_in_range(self, prior: Measurement_t):
        return (prior.mle > self.range[0] and prior.mle < self.range[1])

    def calcDist(self, index):
        if self.fit_type == 'linear':
            estimate = (self.raw_data[index] - self.parameters[0])/self.parameters[1]
            variance = self.model_variance/(self.parameters[1]**2)
        elif self.fit_type == 'hyperbole':
            estimate = self.parameters[0]/(self.raw_data[index]-self.parameters[1])
            variance = self.model_variance/(-self.parameters[1]/estimate**2)**2

        return Measurement_t(estimate, variance)

class KalmanFilter_t:
    def __init__(self):
        self.current_velocity = 0
        self.motion_model = MotionModel_t()
        self.kalman_gains = []

    def update_prior(self, measurement: Measurement_t, velocity_command, time_step):
        if (abs(velocity_command) > abs(self.current_velocity)):
            self.current_velocity = velocity_command * ALPHA + self.current_velocity * (1 - ALPHA)
        else:
            self.current_velocity = velocity_command

        position_estimate = measurement.mle + self.current_velocity*time_step
        position_variance = measurement.variance + self.motion_model.velocity_variance*time_step
        return Measurement_t(position_estimate, position_variance)


    def update_posterior(self, prior: Measurement_t, sensor_measurements: list[Measurement_t]):
        
        kalman_gains = []

        kalman_gain_denom = 1/prior.variance

        #Add the inverse of each variance to the denominator of the Kalman Gain
        for sensor_measurement in sensor_measurements:
            if abs(sensor_measurement.mle - prior.mle) < 4*math.sqrt(sensor_measurement.variance):
                kalman_gain_denom += 1/sensor_measurement.variance
            else:
                sensor_measurements.remove(sensor_measurement)

        #Add the motion model contribution to the position estimate
        position_estimate = (1/prior.variance)/kalman_gain_denom * prior.mle
        kalman_gains.append((1/prior.variance)/kalman_gain_denom)

        #Add each of the sensors contribution to the position estimate
        for sensor_measurement in sensor_measurements:
            position_estimate += (1/sensor_measurement.variance)/kalman_gain_denom * sensor_measurement.mle
            kalman_gains.append((1/sensor_measurement.variance)/kalman_gain_denom)
        
        #Find the position variance
        position_variance = 1/kalman_gain_denom

        posterior = Measurement_t(position_estimate, position_variance)

        if sum(kalman_gains) > 1:
            #Some weird error that only occurs in 10 or so measurements, likely due to rounding
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


        self.sonar1_model = SensorModel_t([-0.017464, 0.99343], 0.00053022, [0, 5], 'linear')
        self.ir3_model = SensorModel_t([0.1361338, 0.2852434], 0.005684, [0.2, 0.7], 'hyperbole')
        self.ir4_model = SensorModel_t([1.25294805, 1.4931097], 0.003980, [1.5, 4], 'hyperbole')

        self.filter = KalmanFilter_t()

        self.sensor_models = []

        self.prior = 0

        self.measurements = []
        self.just_motion = []
        self.just_sensor = []
  

    def time_step(self, index):
        return (self.time[index]-self.time[index-1])

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T
        self.sonar1_model.raw_data = self.sonar1
        self.ir3_model.raw_data = self.raw_ir3
        self.ir4_model.raw_data = self.raw_ir4
        self.sensor_models = [self.sonar1_model, self.ir3_model, self.ir4_model]

    def get_sensor_measurements(self, index):

        return [model.calcDist(index) for model in self.sensor_models if model.check_in_range(self.prior)]

    def run_data(self):
        self.measurements.append(self.sonar1_model.calcDist(0))

        self.just_sensor.append(self.measurements[0])
        self.just_motion.append(self.measurements[0])

        for index in range(1, len(self.time)):
            self.prior = self.filter.update_prior(self.measurements[-1], self.velocity_command[index], self.time_step(index))

            sensor_measurements = self.get_sensor_measurements(index)
            
            self.measurements.append(self.filter.update_posterior(self.prior, sensor_measurements))

            self.just_motion.append(self.filter.update_prior(self.just_motion[-1], self.velocity_command[index], self.time_step(index)))


    def plot_data(self):
        fig, axes = subplots(2)
        fig.suptitle('Motion Model')

        axes[0].plot(self.time, self.sonar2)
        axes[0].plot(self.time, [measurement.mle for measurement in self.measurements])
        
        axes[0].legend(['Sonar2', 'Predicted'])
        

        show()

class TrainingData_t(Data_t):
    """Class that inherits functionality from Data_t but adds distance variable"""
    def __init__(self, filename):
        Data_t.__init__(self, filename)
        self.distance = []

    def load_data(self):
        np_data = np.loadtxt(self.filename, delimiter=',', skiprows=1)
        self.index, self.time, self.distance, self.velocity_command, self.raw_ir1, self.raw_ir2, self.raw_ir3, self.raw_ir4, self.sonar1, self.sonar2 = np_data.T
        self.sonar1_model.raw_data = self.sonar1
        self.ir3_model.raw_data = self.raw_ir3
        self.ir4_model.raw_data = self.raw_ir4
        self.sensor_models = [self.sonar1_model, self.ir3_model, self.ir4_model]

    def plot_data(self):
        fig, axes = subplots(3)
        fig.suptitle('Kalman Filter')

        axes[0].plot(self.time, self.distance)
        axes[0].plot(self.time, [measurement.mle for measurement in self.measurements])
        # axes[0].plot(self.time, [measurement.mle for measurement in self.just_motion])
        axes[0].legend(['Actual', 'Predicted', 'Just Motion'])

        axes[1].set_title("Error")
        axes[1].plot(self.time, self.distance - [measurement.mle for measurement in self.measurements])

        # axes[2].set_title("Kalman Gain")
        # axes[2].plot(self.time[1:], self.filter.kalman_gains)

        show()




def main():
    # data = Data_t('test.csv')
    data = TrainingData_t("training2.csv")
    data.load_data()
    data.run_data()
    data.plot_data()

if __name__ == "__main__":
    main()