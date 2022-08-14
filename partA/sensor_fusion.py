import numpy as np
from matplotlib.pyplot import subplots, show

class Data_t:
    def __init__(self, filename):
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

data = Data_t('test.csv')
data.load_data()

class Kalman_filter_t:
    def __init__(self):
        self.est_velocity = 0
        
    def update_prior(self, velocity_command, current_position, time_step):
        new_velocity = velocity_command
        if (abs(velocity_command) > abs(self.est_velocity)):
            new_velocity -= (velocity_command - self.est_velocity)*0.92
        self.est_velocity = new_velocity
        return current_position + self.est_velocity*time_step