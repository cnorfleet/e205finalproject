import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from ukf import wrap_to_pi, UKFType, UKFWithoutGPSType, N_DOF, N_CONTROL, N_MEAS, N_MEAS_NO_GPS, X_INDEX, Y_INDEX, THETA_INDEX
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

    return (x_gps, y_gps)

def mphToMps(mph):
    return 0.447 * mph

# load data
data = np.genfromtxt("Data/telemetry-v1-2020-03-10-13_50_14.csv", delimiter=",")[270:4300].transpose()
origin = [data[c['Latitude (decimal)']][0], data[c['Longitude (decimal)']][0]]
# print(origin)
(gps_x_all, gps_y_all) = convert_gps_to_xy(data[c['Latitude (decimal)']], data[c['Longitude (decimal)']], origin[0], origin[1])

state = np.zeros((state_dims, 1))
sigma = np.identity(state_dims)

num_samples = data.shape[1]
prev_states_gps = np.zeros((state_dims, num_samples))
prev_variances_gps = np.zeros((state_dims, state_dims, num_samples))
prev_states_no_gps = np.zeros((state_dims, num_samples))
prev_variances_no_gps = np.zeros((state_dims, state_dims, num_samples))

# print('Variances of last 100 points:')
# data_vars = np.std(data[:, -100:], axis=1)
# print(', '.join([f'{list(c.keys())[list(c.values()).index(i)]}:{data_vars[i]:.2f}' for i in range(data_vars.size)]))

# instantiate UKFs
ukfWithGPS = UKFType(N_DOF, N_CONTROL, N_MEAS)
ukfNoGPS = UKFWithoutGPSType(N_DOF, N_CONTROL, N_MEAS_NO_GPS)

# run ukf
for i, measurement in enumerate(data.transpose()):
    # calculate control input
    a_f = measurement[c['Longitudinal Acceleration (m/s^2)']] # m/s^2
    a_r = measurement[c['Lateral Acceleration (m/s^2)']]      # m/s^2
    thetaDot = -1 * measurement[c['Yaw Rate (rad/s)']]        # rad/s
    u_t = np.array([[a_f], [a_r], [thetaDot]])
    R_t = np.diag([0.207010**2, 0.028324**2, 0.000545586**2])
    
    # calculate measurement input
    v_f = mphToMps(measurement[c['Speed (MPH)']])
    gps_x = gps_x_all[i]
    gps_y = gps_y_all[i]
    z_t = np.array([[v_f], [gps_x], [gps_y]])
    Q_t = np.diag([(mphToMps(1))**2, 0.09326**2, 0.11132**2])
    
    # measurement input w/o GPS
    z_t_no_gps = z_t[:-2]
    Q_t_no_gps = Q_t[:-2]
    
    # call both EKF versions on this data
    deltaT = measurement[c['Elapsed Time (ms)']]
    if(not(i == 0 or measurement[c['Lap']] != data[c['Lap']][i-1])):
        deltaT -= data[c['Elapsed Time (ms)']][i-1]
    ukfWithGPS.localize(deltaT, u_t, R_t, z_t, Q_t)
    # ukfNoGPS.localize(deltaT, u_t, R_t, z_t_no_gps, Q_t_no_gps)
    
    # get state and variance
    gps_state = ukfWithGPS.state_est
    gps_sigma = ukfWithGPS.sigma_est
    no_gps_state = ukfNoGPS.state_est
    no_gps_sigma = ukfNoGPS.sigma_est
    
    # update state of without gps ukf to match the ukf with gps unless we're in a simulated gps blackout
    # TODO: aaa
    
    # update variables
    prev_states_gps[:,i] = np.squeeze(gps_state)
    prev_variances_gps[:,:,i] = gps_sigma
    prev_states_no_gps[:,i] = np.squeeze(no_gps_state)
    prev_variances_no_gps[:,:,i] = no_gps_sigma

# Plot data
points =  np.array([gps_x_all, gps_y_all])
plt.figure(1)
plt.plot(points[0], points[1])
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('GPS Position')
plt.figure(3)
plt.plot(data[c['Elapsed Time (ms)']], data[c['Lateral Acceleration (m/s^2)']])
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('GPS Position')

plt.figure(2)
plt.plot(prev_states_gps[X_INDEX], prev_states_gps[Y_INDEX])

# show x, y, theta values
f, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex = True)
line0, = ax0.plot(data[c['Elapsed Time (ms)']], prev_states_gps[X_INDEX, :], color='b')
line1, = ax1.plot(data[c['Elapsed Time (ms)']], prev_states_gps[Y_INDEX, :], color='b')
line2, = ax2.plot(data[c['Elapsed Time (ms)']], prev_states_gps[THETA_INDEX, :], color='b')

gps0,  = ax0.plot(data[c['Elapsed Time (ms)']], gps_x_all, color='r')
gps1,  = ax1.plot(data[c['Elapsed Time (ms)']], gps_y_all, color='r')

# line3, = ax3.plot(TIMES, errors, color='b')
# line3gps, = ax3.plot(TIMES, gps_est_err, color='g')
# ax3.legend((line3, line3gps), ('EKF Error', 'Measurement Model Only Error', 'GPS Error'), loc='upper right')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax3.set_xlabel("Time (seconds)")
ax0.set_ylabel("X Location (meters)")
ax1.set_ylabel("Y Location (meters)")
ax2.set_ylabel("Theta (radians)")
ax3.set_ylabel("Particle Filter Pos Error (m)")
plt.setp(ax0.get_xticklabels(), visible=False)

#fig, ax1 = plt.subplots()
#ax1.set_xlabel('Elapsed Time (ms)')
#ax1.set_ylabel('Steering Angle (deg)', color='tab:blue')
#ax1.plot(data[c['Elapsed Time (ms)']], data[c['Steering Angle (deg)']])
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#color = 'tab:red'
#ax2.set_ylabel('Power Level (KW)', color=color)  # we already handled the x-label with ax1
#ax2.plot(data[c['Elapsed Time (ms)']], data[c['Power Level (KW)']], color=color)
#plt.title('Steering and Power Data')
#fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
