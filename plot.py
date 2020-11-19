import matplotlib.pyplot as plt
import numpy as np

from ukf import wrap_to_pi, UKFType, UKFWithoutGPSType, N_DOF, N_CONTROL, \
                N_MEAS, N_MEAS_NO_GPS, X_INDEX, Y_INDEX, THETA_INDEX, START_ANGLE

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
prev_states_gps_plus3sd = np.zeros((state_dims, num_samples))
prev_states_gps_minus3sd = np.zeros((state_dims, num_samples))

prev_states_no_gps = np.zeros((state_dims, num_samples))
prev_variances_no_gps = np.zeros((state_dims, state_dims, num_samples))
prev_states_no_gps_plus3sd = np.zeros((state_dims, num_samples))
prev_states_no_gps_minus3sd = np.zeros((state_dims, num_samples))

integratedTheta = np.zeros((num_samples))
error_est_with_gps = np.zeros((num_samples))
error_est_no_gps = np.zeros((num_samples))

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
    a_r = -1 * measurement[c['Lateral Acceleration (m/s^2)']] # m/s^2
    thetaDot = measurement[c['Yaw Rate (rad/s)']]        # rad/s
    u_t = np.array([[a_f], [a_r], [thetaDot]])
    R_t = np.diag([(0.023481)**2, (0.027114)**2, (0.000545586*10)**2])
    
    # calculate measurement input
    v_f = mphToMps(measurement[c['Speed (MPH)']])
    gps_x = gps_x_all[i]
    gps_y = gps_y_all[i]
    z_t = np.array([[v_f], [gps_x], [gps_y]])
    Q_t = np.diag([(mphToMps(10))**2, (0.09326)**2, (0.11132)**2])
    
    # measurement input w/o GPS
    z_t_no_gps = z_t[:-2]
    Q_t_no_gps = Q_t[:-2]
    
    # get time delta
    deltaT = measurement[c['Elapsed Time (ms)']]
    if(not(i == 0 or measurement[c['Lap']] != data[c['Lap']][i-1])):
        deltaT -= data[c['Elapsed Time (ms)']][i-1]
    deltaT = deltaT / 1000 # from ms to seconds
    
    # call both EKF versions on this data
    ukfWithGPS.localize(deltaT, u_t, R_t, z_t, Q_t)
    # ukfNoGPS.localize(deltaT, u_t, R_t, z_t_no_gps, Q_t_no_gps)
    
    # get state and variance
    gps_state = ukfWithGPS.state_est
    gps_sigma = ukfWithGPS.sigma_est
    no_gps_state = ukfNoGPS.state_est
    no_gps_sigma = ukfNoGPS.sigma_est
    
    # update variables
    prev_states_gps[:,i] = np.squeeze(gps_state)
    prev_variances_gps[:,:,i] = gps_sigma
    prev_states_no_gps[:,i] = np.squeeze(no_gps_state)
    prev_variances_no_gps[:,:,i] = no_gps_sigma
    error_est_with_gps[i] = np.sqrt((gps_state[X_INDEX, 0] - gps_x)**2 + (gps_state[Y_INDEX, 0] - gps_y)**2)
    error_est_no_gps[i] = np.sqrt((no_gps_state[X_INDEX, 0] - gps_state[X_INDEX, 0])**2 + (no_gps_state[Y_INDEX, 0] - no_gps_state[Y_INDEX, 0])**2)
    if(i == 0):
        integratedTheta[i] = START_ANGLE
    else:
        integratedTheta[i] = wrap_to_pi(integratedTheta[i-1] + thetaDot*deltaT)
        
    for idx in range(N_DOF):
        prev_states_gps_plus3sd[idx][i]  = gps_state[idx] + 3 * np.sqrt(gps_sigma[idx][idx])
        prev_states_gps_minus3sd[idx][i] = gps_state[idx] - 3 * np.sqrt(gps_sigma[idx][idx])
        prev_states_no_gps_plus3sd[idx][i]  = no_gps_state[idx] + 3 * np.sqrt(no_gps_sigma[idx][idx])
        prev_states_no_gps_minus3sd[idx][i] = no_gps_state[idx] - 3 * np.sqrt(no_gps_sigma[idx][idx])
    
    # update state of without gps ukf to match the ukf with gps unless we're in a simulated gps blackout
    if((int(i/500))%2 == 0):
        ukfNoGPS.state_est = ukfWithGPS.state_est
        ukfNoGPS.sigma_est = ukfWithGPS.sigma_est

# Plot data
points =  np.array([gps_x_all, gps_y_all])
plt.figure(1)
plt.plot(prev_states_gps[X_INDEX], prev_states_gps[Y_INDEX], label='Estimated Position')
plt.plot(points[0], points[1], label='GPS Position')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('GPS EKF Position')
plt.legend()
plt.grid(True)
plt.savefig("gps.png", dpi=600)

plt.figure(2)
plt.plot(prev_states_no_gps[X_INDEX], prev_states_no_gps[Y_INDEX], label='Estimated Position')
plt.plot(points[0], points[1], label='GPS Position')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('No GPS EKF Position')
plt.legend()
plt.grid(True)
plt.savefig("nogps.png", dpi=600)


# show x, y, theta values
f, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex = True)
gps0,    = ax0.plot(data[c['Elapsed Time (ms)']]/1000, gps_x_all, color='g')
line0,   = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps[X_INDEX, :], 'b-')
line0p,  = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_plus3sd[X_INDEX, :], 'b:')
line0m,  = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_minus3sd[X_INDEX, :], 'b:')
line0n,  = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps[X_INDEX, :], 'r-')
line0np, = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_plus3sd[X_INDEX, :], 'r:')
line0nm, = ax0.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_minus3sd[X_INDEX, :], 'r:')

gps1,    = ax1.plot(data[c['Elapsed Time (ms)']]/1000, gps_y_all, color='g')
line1,   = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps[Y_INDEX, :], 'b-')
line1p,  = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_plus3sd[Y_INDEX, :], 'b:')
line1m,  = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_minus3sd[Y_INDEX, :], 'b:')
line1n,  = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps[Y_INDEX, :], 'r-')
line1np, = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_plus3sd[Y_INDEX, :], 'r:')
line1nm, = ax1.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_minus3sd[Y_INDEX, :], 'r:')

# gps2,    = ax2.plot(data[c['Elapsed Time (ms)']]/1000, integratedTheta, color='g')
line2,   = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps[THETA_INDEX, :], 'b-')
line2p,  = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_plus3sd[THETA_INDEX, :], 'b:')
line2m,  = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_gps_minus3sd[THETA_INDEX, :], 'b:')
line2n,  = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps[THETA_INDEX, :], 'r-')
line2np, = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_plus3sd[THETA_INDEX, :], 'r:')
line2nm, = ax2.plot(data[c['Elapsed Time (ms)']]/1000, prev_states_no_gps_minus3sd[THETA_INDEX, :], 'r:')

line3g, = ax3.plot(data[c['Elapsed Time (ms)']]/1000, error_est_with_gps, color = 'b')
# line3n, = ax3.plot(data[c['Elapsed Time (ms)']]/1000, error_est_no_gps, color = 'r')
ax0.legend((gps0, line0, line0n), ('GPS', 'EKF with GPS', 'EKF with no GPS'), loc='upper right')
# ax3.legend((line3g, line3n), ('GPS Error', 'EKF with No GPS Error'), loc='upper right')

ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax2.set_xlabel("Time (seconds)")
ax0.set_ylabel("X (meters)")
ax1.set_ylabel("Y (meters)")
ax2.set_ylabel("Theta (radians)")
ax3.set_ylabel("Pos Error (m)")
plt.setp(ax0.get_xticklabels(), visible=False)
plt.savefig("error.png", dpi=600)

plt.figure(10)
plt.plot(data[c['Elapsed Time (ms)']]/1000, data[c['Lateral Acceleration (m/s^2)']])

plt.show()
