"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury

Edits by Sam Bain/Mark Gardyne:
    Line 55: Motion model implementation
    Line 116: Sensor model implementation
"""

from asyncio import wrap_future
import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference

PHI1_STD = 0.005
D_STD = 0.01
PHI2_STD = 0.005

R_STD_BASE = 0.06
PHI_STD_BASE = 0.1
THETA_STD_BASE = 0.15

def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]
    
    #Find the distance travelled since the last measurement
    dx = odom_pose[0] - odom_pose_prev[0]
    dy = odom_pose[1] - odom_pose_prev[1]
    
    #Convert dx and dy to find the most likely estimators for distance d, and angles phi1 and phi2
    phi1_mle = wraptopi(arctan2(dy, dx) - odom_pose_prev[2]) 
    d_mle = np.sqrt(dx**2 + dy**2)
    phi2_mle = angle_difference(arctan2(dy, dx), odom_pose[2])



    for m in range(M):

        #Add a random value to the d, phi1 and phi2 values for each particle to approximate motion model error
        #Standard deviations found using trial and error
        phi1 = phi1_mle + randn(1) * PHI1_STD
        d = d_mle + randn(1) * D_STD
        phi2 = phi2_mle + randn(1) * PHI2_STD

        #initial turn
        particle_poses[m, 2] = wraptopi(particle_poses[m, 2] + phi1)

        #straight travel
        particle_poses[m, 0] += d*cos(particle_poses[m, 2])
        particle_poses[m, 1] += d*sin(particle_poses[m, 2])

        #final turn
        particle_poses[m, 2] = wraptopi(particle_poses[m, 2] + phi2)

    return particle_poses

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)

    #Measure the range and angle to the nearest beacon using sensor measurement
    r = np.sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
    phi = arctan2(beacon_pose[1], beacon_pose[0])

    #Define the standard deviation for the measurements (function of viewing angle) 
    r_std = R_STD_BASE #+ 0.03*(np.pi/2 + beacon_pose[2])**2
    phi_std = PHI_STD_BASE  #+ 0.1*(np.pi/2 + beacon_pose[2])**2

    #Experimenting with beacon angle variable wrt to particle
    beacon_angle_std = THETA_STD_BASE #+ 0.1*(np.pi/2 + beacon_pose[2])**2

    for m in range(M):
        #Get the particles pose from the list of poses
        particle_pose = particle_poses[m, :]

        #Find the relevant measurements given the particle pose and beacon location
        r_particle = np.sqrt((beacon_loc[0]-particle_pose[0])**2 + (beacon_loc[1]-particle_pose[1])**2)
        phi_particle = angle_difference(particle_pose[2], arctan2(beacon_loc[1] - particle_pose[1], beacon_loc[0] - particle_pose[0]))
        beacon_angle_particle = wraptopi(particle_pose[2] + beacon_pose[2])

        #Determine the likihood of the given measurements for the particle 
        range_likelihood = gaussian(r - r_particle, 0, r_std)
        phi_likelihood = gaussian(angle_difference(phi, phi_particle), 0, phi_std)

        beacon_angle_likelihood = gaussian(angle_difference(beacon_loc[2], beacon_angle_particle), 0, beacon_angle_std)

        #Update the weight given the likelihoods
        particle_weights[m] = range_likelihood * phi_likelihood * beacon_angle_likelihood

    return particle_weights
