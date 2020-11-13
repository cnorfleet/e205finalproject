import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv

from ukf import wrap_to_pi, UKFType, UKFWithoutGPSType, N_DOF, N_CONTROL, N_MEAS, N_MEAS_NO_GPS
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

# load data
data = np.genfromtxt("Data/telemetry-v1-2020-03-10-13_50_14.csv", delimiter=",")[270:4300].transpose()
origin = [data[c['Latitude (decimal)']][0], data[c['Longitude (decimal)']][0]]
(gps_x_all, gps_y_all) = convert_gps_to_xy(data[c['Latitude (decimal)']], data[c['Longitude (decimal)']], origin[0], origin[1])

print('Variances of last 100 points:')
data_vars = np.std(data[:, -100:], axis=1)
print(', '.join([f'{list(c.keys())[list(c.values()).index(i)]}:{data_vars[i]:.2f}' for i in range(data_vars.size)]))
