import numpy as np
from numpy import sin, cos
import math
from scipy.linalg import sqrtm

START_ANGLE = 5*np.pi/6 #-2*np.pi/3

X_INDEX = 0
Y_INDEX = 1
THETA_INDEX = 2
XDOT_INDEX = 3
YDOT_INDEX = 4
N_DOF = 5

AF_INPUT_INDEX = 0
AR_INPUT_INDEX = 1
THETADOT_INPUT_INDEX = 2
N_CONTROL = 3

# Note: GPS moved to last elements to unify between GPS and no GPS UKFs
VF_MEAS_INDEX = 0
GPS_X_MEAS_INDEX = 1
GPS_Y_MEAS_INDEX = 2
N_MEAS = 3
N_MEAS_NO_GPS = 1

def wrap_to_pi(angle):
    while angle >= math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

class UKFBaseType:
    def __init__(self, nDOF, nControl, nMeas, scaling_factor = None):
        if(scaling_factor == None):
            scaling_factor = nDOF - 3
        self.nDOF = nDOF
        self.nControl = nControl
        self.nMeas = nMeas
        self.state_est = np.array([[0], [0], [wrap_to_pi(START_ANGLE)], [0], [0]])
        self.sigma_est = np.diag([1**2, 1**2, 0.1**2, 10**2, 10**2]) # initially large for arbitrary estimate
        self.scaling_factor = scaling_factor # lambda scaling factor
        
    def getSigmaPoints(state, sigma_matrix, nDOF, scaling_factor):
        sigmaPts = np.zeros((1+nDOF*2, nDOF, 1))
        sigmaPts[0] = state
        for i in range(nDOF):
            diff = (sqrtm((nDOF + scaling_factor) * sigma_matrix))[i]
            sigmaPts[i+1]      = (state + [[diff[j]] for j in range(nDOF)])
            sigmaPts[i+1,      THETA_INDEX, 0] = wrap_to_pi(sigmaPts[i+1,      THETA_INDEX, 0])
            sigmaPts[i+1+nDOF] = (state - [[diff[j]] for j in range(nDOF)])
            sigmaPts[i+1+nDOF, THETA_INDEX, 0] = wrap_to_pi(sigmaPts[i+1+nDOF, THETA_INDEX, 0])
        return sigmaPts

    def getWeights(self):
        # weights for the regrouping algorithm.  note that the sum of all weights is 1
        weights = [ self.scaling_factor / (self.nDOF + self.scaling_factor) ]
        for i in range(2 * self.nDOF):
            weights = weights + [1 / (2 * (self.nDOF + self.scaling_factor))]
        return weights

    def get_G_u_t(self, dt, state_est, u_t):
        manual = self.get_G_u_t_manual(dt, state_est, u_t)
        # auto = diff_function(self.applyMotionModelSingle, [dt, state_est, u_t], param=2)
        return manual

    def regroupSigmaPoints(self, dt, sigma_points_pred, u_t, R_t):
        # state prediction
        weights = self.getWeights()
        state_pred = np.zeros((self.nDOF,1))
        for i in range(2 * self.nDOF + 1):
            sigma_points_pred[i][THETA_INDEX, 0] = wrap_to_pi(sigma_points_pred[i][THETA_INDEX, 0])
            state_pred = state_pred + weights[i] * sigma_points_pred[i]
        state_pred[THETA_INDEX, 0] = wrap_to_pi(state_pred[THETA_INDEX, 0])
            
        # update covariance matrix
        sigma_pred = np.zeros((self.nDOF, self.nDOF))
        for i in range(2 * self.nDOF + 1):
            diff = sigma_points_pred[i] - state_pred
            sigma_pred = sigma_pred + weights[i] * (diff @ diff.T)
        G_u_t = self.get_G_u_t(dt, self.state_est, u_t)
        sigma_pred = sigma_pred + G_u_t @ R_t @ G_u_t.T
            
        return (state_pred, sigma_pred)

    def getPredictedMeasurements(self, sigma_points):
        Z_t_pred = []
        for sigma_pt in sigma_points:
            z_t = self.get_z_pred(sigma_pt)
            Z_t_pred.append(z_t)
        return np.array(Z_t_pred)

    def correctionStep(self, state_pred, sigma_pred, sigma_points_pred, z_t, Q_t):
        Z_t_pred = self.getPredictedMeasurements(sigma_points_pred)
        weights = self.getWeights()
        z_t_pred = np.zeros((self.nMeas, 1))
        for i in range(2*self.nDOF+1):
            z_t_pred = z_t_pred + weights[i] * Z_t_pred[i]
        
        S_t = np.zeros((self.nMeas, self.nMeas))
        for i in range(2*self.nDOF+1):
            diff = Z_t_pred[i] - z_t_pred
            S_t = S_t + weights[i] * (diff @ diff.T)
        S_t = S_t + Q_t
        
        sigma_x_z_t = np.zeros((self.nDOF, self.nMeas))
        for i in range(2*self.nDOF+1):
            state_diff = sigma_points_pred[i] - state_pred
            state_diff[THETA_INDEX, 0] = wrap_to_pi(state_diff[THETA_INDEX, 0])
            meas_diff = Z_t_pred[i] - z_t_pred
            sigma_x_z_t = sigma_x_z_t + weights[i] * (state_diff @ meas_diff.T)
        
        K_t = sigma_x_z_t @ np.linalg.inv(S_t)
        state_est = state_pred + K_t @ (z_t - z_t_pred)
        state_est[THETA_INDEX, 0] = wrap_to_pi(state_est[THETA_INDEX, 0]) # clip theta
        sigma_est = sigma_pred - (K_t @ S_t @ K_t.T)
        
        self.state_est = state_est
        self.sigma_est = sigma_est

    def applyMotionModel(self, dt, sigma_points, u_t):
        # predict next state for each sigma point
        sigma_points_pred = []
        for sigma_pt in sigma_points:
            pt = self.applyMotionModelSingle(dt, sigma_pt, u_t)
            sigma_points_pred.append(pt)
        return sigma_points_pred

    def applyMotionModelSingle(self, dt, sigma_pt, u_t):
        pt = np.zeros((5,1))
        pt[X_INDEX, 0]     = sigma_pt[X_INDEX] + sigma_pt[XDOT_INDEX] * dt
        pt[Y_INDEX, 0]     = sigma_pt[Y_INDEX] + sigma_pt[YDOT_INDEX] * dt
        pt[THETA_INDEX, 0] = wrap_to_pi(sigma_pt[THETA_INDEX] + u_t[THETADOT_INPUT_INDEX] * dt)
        pt[XDOT_INDEX, 0]  = sigma_pt[XDOT_INDEX] + dt * (u_t[AF_INPUT_INDEX] * cos(sigma_pt[THETA_INDEX]) + u_t[AR_INPUT_INDEX] * sin(sigma_pt[THETA_INDEX]))
        pt[YDOT_INDEX, 0]  = sigma_pt[YDOT_INDEX] + dt * (u_t[AF_INPUT_INDEX] * sin(sigma_pt[THETA_INDEX]) - u_t[AR_INPUT_INDEX] * cos(sigma_pt[THETA_INDEX]))
        return pt

    def get_G_u_t_manual(self, dt, state_est, u_t):
        return np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, dt],
                         [cos(state_est[THETA_INDEX, 0]) * dt,      sin(state_est[THETA_INDEX, 0]) * dt, 0],
                         [sin(state_est[THETA_INDEX, 0]) * dt, -1 * cos(state_est[THETA_INDEX, 0]) * dt, 0]])
    
    def localize(self, dt, u_t, R_t, z_t, Q_t):
        # prediction step using sigma points
        sigma_points = UKFBaseType.getSigmaPoints(self.state_est, self.sigma_est, self.nDOF, self.scaling_factor)
        sigma_points_model = self.applyMotionModel(dt, sigma_points, u_t)
        (state_pred, sigma_pred) = self.regroupSigmaPoints(dt, sigma_points_model, u_t, R_t)
        sigma_points_pred = UKFBaseType.getSigmaPoints(state_pred, sigma_pred, self.nDOF, self.scaling_factor)
        
        # correction step
        outputs = self.correctionStep(state_pred, sigma_pred, sigma_points_pred, z_t, Q_t)
        
        return outputs


class UKFType(UKFBaseType):
    def get_z_pred(self, sigma_pt):
        return np.array([[sigma_pt[XDOT_INDEX, 0] * cos(sigma_pt[THETA_INDEX, 0]) +
                          sigma_pt[YDOT_INDEX, 0] * sin(sigma_pt[THETA_INDEX, 0])],
                         [sigma_pt[X_INDEX, 0]],
                         [sigma_pt[Y_INDEX, 0]]])

class UKFWithoutGPSType(UKFBaseType):
    def get_z_pred(self, sigma_pt):
        return np.array([[sigma_pt[XDOT_INDEX, 0] * cos(sigma_pt[THETA_INDEX, 0]) +
                          sigma_pt[YDOT_INDEX, 0] * sin(sigma_pt[THETA_INDEX, 0])]])

def diff_function(func, params, param = 0):
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
