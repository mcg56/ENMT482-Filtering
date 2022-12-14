"""Particle filter demonstration program.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury

Note, this example is written in an attempt to be straightforward
to follow for those new to Python.  Normally, we would have written
it using about a dozen classes.

Edits by Sam Bain/Mark Gardyne:
    Line 105: Changed initial number of particles
    Line 109: Made robots initial position unknown
    Line 173: When robot is lost, spread poses back out across map.
    Line 181: Recalculating required number of particles based on the correlation between particle poses and sensor measurements.
"""

from __future__ import print_function, division
from numpy.random import uniform, seed
try:
    import matplotlib; matplotlib.use("TkAgg")        
except:
    import matplotlib; matplotlib.use("Qt5Agg")

from models import motion_model, sensor_model
from utils import *
from plot import *
from transform import *
from matplotlib.ticker import PercentFormatter
import numpy as np
import time

#Change seed to experiment with different scenarios
seed(7)

# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta,
# beacon_ids, beacon_x, beacon_y, beacon_theta
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# Time in ns
t = data[:, 0]

# Velocity command in m/s, rotation command in rad/s
commands = data[:, 1:3]

# Position in map frame, from SLAM (this approximates ground truth)
slam_poses = data[:, 3:6]

# Position in odometry frame, from wheel encoders and gyro
odom_poses = data[:, 6:9]


# Id and measured position of beacon in camera frame
beacon_ids = data[:, 9]
beacon_poses = data[:, 10:13]
# Use beacon id of -1 if no beacon detected
beacon_ids[np.isnan(beacon_ids)] = -1
beacon_ids = beacon_ids.astype(int)
beacon_visible = beacon_ids >= 0

# map_data is a 16x13 matrix.  Its columns are:
# beacon_ids, x, y, theta, (9 columns of covariance)
map_data = np.genfromtxt('beacon_map.csv', delimiter=',', skip_header=1)

Nbeacons = map_data.shape[0]
beacon_locs = np.zeros((Nbeacons, 3))
for m in range(Nbeacons):
    id = int(map_data[m, 0])
    beacon_locs[id] = map_data[m, 1:4]

# Remove jumps in the pose history
slam_poses = clean_poses(slam_poses)

# Transform odometry poses into map frame
odom_to_map = find_transform(odom_poses[0], slam_poses[0])
odom_poses = transform_pose(odom_to_map, odom_poses)

plt.ion()
fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)
fig.canvas.mpl_connect('key_press_event', keypress_handler)
# fig.canvas.manager.full_screen_toggle()

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, '-', label='SLAM')
# Uncomment to show odometry when debugging
# plot_path(axes, odom_poses, 'b:', label='Odom')

axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
axes.figure.canvas.draw()
axes.figure.canvas.flush_events()

start_step = 0

Nparticles = 2000

display_steps = 10

#Unknown initial position
Xmin = min(slam_poses[:, 0])
Xmax = max(slam_poses[:, 0])
Ymin = min(slam_poses[:, 1])
Ymax = max(slam_poses[:, 1])

Tmin = -np.pi
Tmax = np.pi

weights = np.ones(Nparticles)
poses = np.zeros((Nparticles, 3))

for m in range(Nparticles):
    poses[m] = (uniform(Xmin, Xmax),
                uniform(Ymin, Ymax),
                uniform(Tmin, Tmax))

initial_rand_poses = poses

Nposes = odom_poses.shape[0]
print(Nposes)
est_poses = np.zeros((Nposes, 3))

plot_particles(axes, poses, weights)
axes.set_title('Push space to start/stop, dot to move one step, q to quit...')
# wait_until_key_pressed()


state = 'run'
display_step_prev = 0

robot_capture_index = 200
robot_release_index = 400

    
error_array = np.array([])
viewing_angle_array = np.array([])

exec_time_array = np.array([])

#Kidnapped robot
# for n in list(range(start_step+1, robot_capture_index)) + list(range(robot_release_index, Nposes)):


for n in range(start_step + 1, Nposes):

    poses = motion_model(poses, commands[n-1], odom_poses[n], odom_poses[n - 1],
                         t[n] - t[n - 1])

    
    
    #Beacon 4 provided some inaccurate readings, so was removed
    if beacon_visible[n] and beacon_ids[n] != 4:


        beacon_id = beacon_ids[n]
        beacon_loc = beacon_locs[beacon_id]
        beacon_pose = beacon_poses[n]

        weight_likelihoods = sensor_model(poses, beacon_pose, beacon_loc)

        weights *= weight_likelihoods

        if sum(weights) < 1e-50:
            #Robot is lost, so spread particles back out across entire area
            weights = np.ones(Nparticles)

            poses = initial_rand_poses
            

        if is_degenerate(weights):
            # print('Resampling %d' % n)
            num_of_particles = round(min(2000, 200/max(weight_likelihoods)))
            # print(num_of_particles)     
            poses, weights = resample(poses, weights, num_of_particles)

    est_poses[n] = poses.mean(axis=0)

    if (n > display_step_prev + display_steps) or state == 'step':
        # print(n)

        # Show particle cloud
        plot_particles(axes, poses, weights)

        # Leave breadcrumbs showing current odometry
        # plot_path(axes, odom_poses[n], 'k.')

        # Show mean estimate
        plot_path_with_visibility(axes, est_poses[display_step_prev-1 : n+1],
                                  '-', visibility=beacon_visible[display_step_prev-1 : n+1])
        display_step_prev = n

        # print(state)
        
    # key = get_key()
    # if key == '.':
    #     state = 'step'
    # elif key == ' ':
    #     if state == 'run':
    #         state = 'pause'
    #     else:
    #         state = 'run'

    # if state == 'pause':
    #     wait_until_key_pressed()
    # elif state == 'step':
    #     wait_until_key_pressed()            


# # Display final plot
print('Done, displaying final plot')
plt.ioff()
plt.show()

# Save final plot to file
plot_filename = 'path.pdf'
print('Saving final plot to', plot_filename)

plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

fig = plt.figure(figsize=(10, 5))
axes = fig.add_subplot(111)

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, 'b-', label='SLAM')
plot_path(axes, odom_poses, 'b:', label='Odom')
plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
fig.savefig(plot_filename, bbox_inches='tight')
