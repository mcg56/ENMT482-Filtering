import numpy as np
from matplotlib.pyplot import *
import math

ALPHA = 0.035

CUNT = []

class Measurement_t:
    def __init__(self, mle, variance):
        self.mle = mle
        self.variance = variance

class MotionModel_t:
    def __init__(self):
        self.velocity_variance = 0.001178525902611975

class SonarModel_t:
    def __init__(self):
        self.parameters = [-0.017464, 0.99343]
        self.model_variance = 0.00053022

    def calcDist(self, sensor_measurement, previous_mle):
        estimate = (sensor_measurement - self.parameters[0])/self.parameters[1]
        if (abs(previous_mle-estimate) > 0.5):
            variance = 1000
        else:
            variance = self.model_variance/(self.parameters[1]**2)
        return Measurement_t(estimate, variance)

class IR3Model_t:
    def __init__(self):
        self.parameters = [[0.1361338, 0.2852434], [0.28991766, 0.11192217]]
        self.model_variance = [0.005683673621942591, 0.005847994008104537]

    def calcDist(self, sensor_measurement, previous_mle):
        mle_array = []
        difference_array = []

        mle_array.append(self.parameters[0][1]/(sensor_measurement-self.parameters[0][0]))
        mle_array.append((sensor_measurement - self.parameters[1][0])/self.parameters[1][1])

        difference_array.append(abs(previous_mle- mle_array[0]))
        difference_array.append(abs(previous_mle- mle_array[1]))

        mle_index = difference_array.index(min(difference_array))
        estimate = mle_array[mle_index]
        variance = 0
        return Measurement_t(estimate, variance)



    # def inverse_piecewise(sensor, inverse_eqn_array, params_array, prev_X_blue):
    #     MLE_array = []
    #     diff_array = []

    #     for i in range(0, len(inverse_eqn_array)):
    #         try:
    #             X_mle = inverse_eqn_array[i](z, params_array[i])
    #             MLE_array.append(X_mle)
    #             diff_array.append(abs(prev_X_blue - X_mle))
    #         except Exception as E:
    #             print("Could not invert piecewise equation {} for z = {}".format(i, z))
    #     smallest_diff_index = diff_array.index(min(diff_array))
    #     return MLE_array[smallest_diff_index], smallest_diff_index


class IR4Model_t:
    def __init__(self):
        self.parameters = [[2.1989776 , -5.93744738, 27.598602], [3.43724799, 0.02907117], [1.25294805, 1.4931097]]
        self.model_variance = [0.003065322291272046, 0.006230583811321369, 0.003980472133529135]

    def calcDist(self, sensor_measurement, previous_mle):
        estimate = 0
        variance = 0
        return Measurement_t(estimate, variance)

class SensorModels_t:
    def __init__(self):
        self.sonar1_model = SonarModel_t()
        self.ir3_model = IR3Model_t()
        self.ir4_model = IR4Model_t()

    def fuse_sensors(self, sonar1_meas, prior):
        return self.sonar1_model.calcDist(sonar1_meas, prior) 


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
        position_variance = measurement.variance + self.motion_model.velocity_variance*time_step**2
        return Measurement_t(position_estimate, position_variance)


    def update_posterior(self, prior: Measurement_t, sensor_measurement: Measurement_t):
        if abs(sensor_measurement.mle - prior.mle) < 4*math.sqrt(sensor_measurement.variance):
            kalman_gain = (1/sensor_measurement.variance)/(1/prior.variance + 1/sensor_measurement.variance)
            position_estimate = kalman_gain * sensor_measurement.mle + (1-kalman_gain) * prior.mle
            position_variance = 1/(1/prior.variance + 1/sensor_measurement.variance)
            posterior = Measurement_t(position_estimate, position_variance)
        else:
            kalman_gain = 0
            posterior = prior

        #Append kalman gains to list to plot    
        self.kalman_gains.append(kalman_gain)

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
        self.measurements.append(self.sensors_models.sonar1_model.calcDist(self.sonar1[0], self.sonar1[0]))

        self.just_sensor.append(self.measurements[0])
        self.just_motion.append(self.measurements[0])

        for index in range(1, len(self.time)):
            self.priors.append(self.filter.update_prior(self.measurements[-1], self.velocity_command[index], self.time_step(index)))
            # self.measurements.append(self.filter.update_posterior(self.priors[-1], self.sensors_models.sonar1_model.calcDist(self.sonar1[index], self.priors[-1].mle)))
            global CUNT
            CUNT.append(self.sensors_models.ir3_model.calcDist(self.raw_ir3[index],self.priors[-1].mle).mle)
            self.measurements.append(self.filter.update_posterior(self.priors[-1], self.sensors_models.fuse_sensors(self.sonar1[index], self.priors[-1].mle)))

            self.just_motion.append(self.filter.update_prior(self.just_motion[-1], self.velocity_command[index], self.time_step(index)))
            self.just_sensor.append(self.sensors_models.sonar1_model.calcDist(self.sonar1[index], self.priors[-1].mle))


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
        fig, axes = subplots(4)
        fig.suptitle('Kalman Filter')

        axes[0].plot(self.time, self.distance)
        # axes[0].plot(self.time, [measurement.mle for measurement in self.just_sensor])
        axes[0].plot(self.time, [measurement.mle for measurement in self.measurements])
        # axes[0].plot(self.time, [measurement.mle for measurement in self.just_motion])
        axes[0].legend(['Actual', 'Predicted', 'Just Motion'])

        axes[1].set_title("Error")
        axes[1].plot(self.time, self.distance - [measurement.mle for measurement in self.measurements])

        axes[2].set_title("Kalman Gain")
        axes[2].plot(self.time[1:], self.filter.kalman_gains)

        axes[3].set_title("Kalman Gain")
        axes[3].plot(self.time[1:], CUNT)


        show()




def main():
    # data = Data_t('test.csv')
    data = TrainingData_t("training1.csv")
    data.load_data()
    data.run_data()
    data.plot_data()

if __name__ == "__main__":
    main()