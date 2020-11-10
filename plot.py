import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv

# data column lookup
c = {key: i for i, key in enumerate(['Lap','Elapsed Time (ms)','Speed (MPH)','Latitude (decimal)','Longitude (decimal)','Lateral Acceleration (m/s^2)','Longitudinal Acceleration (m/s^2)','Throttle Position (%)','Brake Pressure (bar)','Steering Angle (deg)','Steering Angle Rate (deg/s)','Yaw Rate (rad/s)','Power Level (KW)','State of Charge (%)','Tire Pressure Front Left (bar)','Tire Pressure Front Right (bar)','Tire Pressure Rear Left (bar)','Tire Pressure Rear Right (bar)','Brake Temperature Front Left (% est.)','Brake Temperature Front Right (% est.)','Brake Temperature Rear Left (% est.)','Brake Temperature Rear Right (% est.)','Front Inverter Temp (%)','Rear Inverter Temp (%)','Battery Temp (%)','Tire Slip Front Left (% est.)','Tire Slip Front Right (% est.)','Tire Slip Rear Left (% est.)','Tire Slip Rear Right (% est.)'])}
# state column lookup
s = {key: i for i, key in enumerate(['x','y','z','t'])}
state_dims = len(s)

def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    EARTH_RADIUS = 6.3781E6  # meters
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(np.pi/180.)*(lon_gps - lon_origin)*np.cos((np.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(np.pi/180.)*(lat_gps - lat_origin)

    return np.array([x_gps, y_gps])

def main():
    # load data
    data = np.genfromtxt("telemetry-v1-2020-03-10-13_50_14.csv", delimiter=",")[270:4300].transpose()
    print(data.shape)
    origin = [data[c['Latitude (decimal)']][0], data[c['Longitude (decimal)']][0]]
    print(origin)
    xy = convert_gps_to_xy(data[c['Latitude (decimal)']], data[c['Longitude (decimal)']], origin[0], origin[1])

    # state = [x, y, z, t]
    x = np.zeros((state_dims, 1))
    sigma = np.identity(state_dims)

    num_samples = data.shape[1]
    prev_states = np.zeros((state_dims, num_samples))
    prev_variances = np.zeros((state_dims, state_dims, num_samples))
    prev_corrections = np.zeros((state_dims, num_samples))

    print('Variances of last 100 points:')
    data_vars = np.std(data[:, -100:], axis=1)
    print(', '.join([f'{list(c.keys())[list(c.values()).index(i)]}:{data_vars[i]:.2f}' for i in range(data_vars.size)]))

    # run ekf
    if False: #for i, measurement in enumerate(data.transpose()):
        # prediction step
        control_vector = np.array([measurement[c['u']], measurement[c['l']], measurement[c['r']]])

        control_vector_sigma = np.diag([0.22, 0.22, 0.22])   # from last 100 points
        x_bar = propogate_state(x, control_vector)

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



def angle_wrap(angle):
    angle = angle % (2*np.pi)
    if angle > np.pi:
        angle -= np.pi*2
    return angle

def measure(m):
    """Convert CSV to measurement."""
    return np.array([m[c['x']], m[c['y']], m[c['z']], m[c['theta']]]).reshape(-1, 1)

def sensor_model(x):
    """Convert state to predicted sensor measurement."""
    # just return the x, y, z, theta of the state
    return np.array([x[s['x']], x[s['y']], x[s['z']], x[s['t']]]).reshape(-1, 1)

def diff_function(params, func, param=0):
    """Takes derivative of Python function func at location params about parameter param.

    I didn't want to take derivatives, so I spent far longer working on this function than if I had just taken them. I
    still took the derivatives to verify that it works, and it seems to be very accurate (same to all decimals printed
    where tested). This assumes that the function is decently well-behaved and doesn't have any weird jumps.
    """
    # compute reference point
    a = func(*params)

    # output array is modified array by output
    result = np.zeros((a.size, params[param].size))
    for i in range(params[param].size):
        # twiddle each parameter
        delta = np.zeros(params[param].shape)
        twiddle = 0.00001   # named in honor of the twiddle factor in Sebastian Thrun's intro
        delta[i] = twiddle  # to robotics class PID autotuner
        
        # construct function arguments
        param_d = params[:param] + [params[param] + delta] + params[param+1:]
        b = func(*param_d)
        derivative = (b-a)/twiddle
        result[:, i] = np.squeeze(derivative)
    return result

def propogate_state(x_prev, cv, dt=0.1):
    """Apply nonlinear state propagation.
    All jacobians are calculated from here, so this and the columns are all which must be changed
    to adapt the EKF to a new plant."""

    x_prev = np.squeeze(x_prev)  # just to make numpy nicer

    # blimp frame
    s_ = 0.5*cv[0]               # up/down
    v = 1.0*(cv[1]+cv[2])        # forward/backward
    w = 0.315*(-cv[1]+cv[2])     # theta

    # transform to global frame
    ax = np.cos(x_prev[s['t']])*v
    ay = np.sin(x_prev[s['t']])*v
    az = s_

    # state = [x, y, z, t]
    x = [
        # positions
        x_prev[0] + dt*ax,
        x_prev[1] + dt*ay,
        x_prev[2] + dt*az,

        # angles
        x_prev[s['t']] + w*dt,
    ]
    return np.array(x).reshape(-1, 1)


if __name__ == "__main__":
    main()