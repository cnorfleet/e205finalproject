import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv

from ukf import *

# data column lookup
c = {key: i for i, key in enumerate(['Lap',
                                     'Elapsed Time (ms)',
                                     'Speed (MPH)',
                                     'Latitude (decimal)',
                                     'Longitude (decimal)',
                                     'Lateral Acceleration (m/s^2)',
                                     'Longitudinal Acceleration (m/s^2)',
                                     'Throttle Position (%)',
                                     'Brake Pressure (bar)',
                                     'Steering Angle (deg)',
                                     'Steering Angle Rate (deg/s)',
                                     'Yaw Rate (rad/s)',
                                     'Power Level (KW)',
                                     'State of Charge (%)',
                                     'Tire Pressure Front Left (bar)',
                                     'Tire Pressure Front Right (bar)',
                                     'Tire Pressure Rear Left (bar)',
                                     'Tire Pressure Rear Right (bar)',
                                     'Brake Temperature Front Left (% est.)',
                                     'Brake Temperature Front Right (% est.)',
                                     'Brake Temperature Rear Left (% est.)',
                                     'Brake Temperature Rear Right (% est.)',
                                     'Front Inverter Temp (%)',
                                     'Rear Inverter Temp (%)',
                                     'Battery Temp (%)',
                                     'Tire Slip Front Left (% est.)',
                                     'Tire Slip Front Right (% est.)',
                                     'Tire Slip Rear Left (% est.)',
                                     'Tire Slip Rear Right (% est.)'])}
# state column lookup
s = {key: i for i, key in enumerate(['x','y','t','xd','yd'])}
state_dims = len(s)

def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    EARTH_RADIUS = 6.3781E6  # meters
    """Convert gps coordinates to cartesian with equirectangular projection"""
    x_gps = EARTH_RADIUS*(np.pi/180.)*(lon_gps - lon_origin)*np.cos((np.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(np.pi/180.)*(lat_gps - lat_origin)

    return np.array([x_gps, y_gps])

def measure(m):
    """Convert CSV to measurement."""
    return np.array([m[c['x']], m[c['y']], m[c['z']], m[c['theta']]]).reshape(-1, 1)

def sensor_model(x):
    """Convert state to predicted sensor measurement."""
    # just return the x, y, z, theta of the state
    return np.array([x[s['x']], x[s['y']], x[s['z']], x[s['t']]]).reshape(-1, 1)

# load data
data = np.genfromtxt("Data/telemetry-v1-2020-03-10-13_50_14.csv", delimiter=",")[270:4300].transpose()
origin = [data[c['Latitude (decimal)']][0], data[c['Longitude (decimal)']][0]]
print(origin)
xy = convert_gps_to_xy(data[c['Latitude (decimal)']], data[c['Longitude (decimal)']], origin[0], origin[1])

state = np.zeros((state_dims, 1))
sigma = np.identity(state_dims)

num_samples = data.shape[1]
prev_states = np.zeros((state_dims, num_samples))
prev_variances = np.zeros((state_dims, state_dims, num_samples))
prev_corrections = np.zeros((state_dims, num_samples))

# print('Variances of last 100 points:')
# data_vars = np.std(data[:, -100:], axis=1)
# print(', '.join([f'{list(c.keys())[list(c.values()).index(i)]}:{data_vars[i]:.2f}' for i in range(data_vars.size)]))

# run ukf
for i, measurement in enumerate(data.transpose()):
    # calculate control input
    a_f = measurement[c['Longitudinal Acceleration (m/s^2)']] # m/s^2
    a_r = measurement[c['Lateral Acceleration (m/s^2)']]      # m/s^2
    thetaDot = -1 * measurement[c['Yaw Rate (rad/s)']]        # rad/s
    u_t = np.matrix([[a_f], [a_r], [thetaDot]])
    R_t = np.diag([1**2, 1**2, 1**2])   # u_t variance, from last 100 points # TODO: make these values right
    
    # calculate measurement input
    v_f = measurement[c['Speed (MPH)']]
    gps_x = 0
    gps_y= 0
    z_t = np.matrix([[v_f], [gps_x], [gps_y]])
    Q_t = np.diag([1**2, 1**2, 1**2])
    
    #x_bar = propogate_state(state, control_vector)

    # Call our derivative function to differentiate the sensor model
    # Take derivative of state update with respect to previous state
    G_x_t = diff_function([x, control_vector], propogate_state, param=0)
    # Take derivative of state update with respect to control input
    G_u_t = diff_function([x, control_vector], propogate_state, param=1)

    # update EKF sigma
    sigma_bar = G_x_t @ sigma @ G_x_t.transpose() + \
                G_u_t @ control_vector_sigma @ G_u_t.transpose()

    # correction step
    # differentiate sensor model
    h_t = diff_function([x], sensor_model, param=0)
    sensor_sigma = np.diag([0.8, 0.8, 0.8, 0.1])   # from last 100 points
    z_t = measure(measurement)

    # calculate Kalman gain
    k_t = sigma_bar @ h_t.transpose()@inv(h_t@sigma_bar@h_t.transpose()+sensor_sigma)
    error = z_t - sensor_model(x_bar)
    error[3] = angle_wrap(error[3])

    correction = k_t.dot(error)
    x_est = x_bar + correction
    x_est[s['t']] = angle_wrap(x_est[s['t']])

    sigma_est = (np.identity(k_t.shape[0])-k_t.dot(h_t)).dot(sigma_bar)
    
    # update variables
    x = x_est
    sigma = sigma_est
    prev_states[:,i] = np.squeeze(x)
    prev_corrections[:,i] = np.squeeze(correction)
    prev_variances[:,:,i] = sigma

# Plot data
points = xy
plt.figure(1)
plt.plot(points[0], points[1])
print(xy)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('GPS Position')

plt.figure(2)
plt.plot(data[c['Yaw Rate (rad/s)']], data[c['Steering Angle (deg)']]*data[c['Speed (MPH)']])

fig, ax1 = plt.subplots()

ax1.set_xlabel('Elapsed Time (ms)')
ax1.set_ylabel('Steering Angle (deg)', color='tab:blue')
ax1.plot(data[c['Elapsed Time (ms)']], data[c['Steering Angle (deg)']])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Power Level (KW)', color=color)  # we already handled the x-label with ax1
ax2.plot(data[c['Elapsed Time (ms)']], data[c['Power Level (KW)']], color=color)
plt.title('Steering and Power Data')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
