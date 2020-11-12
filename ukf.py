import numpy as np
import math
from scipy.linalg import sqrtm

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

def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle <= -math.pi:
        angle += 2*math.pi
    return angle

class UKFBaseType:
    def __init__(self, nDOF, nControl, nMeas, scaling_factor = nDOF-3):
        self.nDOF = nDOF
        self.nControl = nControl
        self.nMeas = nMeas
        self.state_est = np.matrix([[0] for _ in range(nDOF)])
        self.sigma_est = np.diag([100 for _ in range(nDOF)]) # initially large for arbitrary estimate
        self.scaling_factor = scaling_factor # lambda scaling factor
        
    def getSigmaPoints(state, sigma_matrix, nDOF, scaling_factor):
        sigma_0 = state
        sigma_1_n = []
        for i in range(nDOF):
            diff = (sqrtm((nDOF + scaling_factor) * np.array(sigma_matrix)))[i]
            diff = np.matrix(diff)
            sigma_pt = state + [[diff[0, i]] for i in range(nDOF)]
            sigma_1_n = sigma_1_n + [sigma_pt]
        sigma_nplus1_2n = []
        for j in range(nDOF):
            i = j + nDOF
            diff = (sqrtm((nDOF + scaling_factor) * sigma_matrix))[i-nDOF]
            diff = np.matrix(diff)
            sigma_pt = state - [[diff[0, i]] for i in range(nDOF)]
            sigma_nplus1_2n = sigma_nplus1_2n + [sigma_pt]
        return [sigma_0] + sigma_1_n + sigma_nplus1_2n
    
    def getWeights(self):
        # weights for the regrouping algorithm.  note that the sum of all weights is 1
        weights = [ self.scaling_factor / (self.nDOF + self.scaling_factor) ]
        for i in range(2 * self.nDOF):
            weights = weights + [1 / (2 * (self.nDOF + self.scaling_factor))]
        return weights
    
    def regroupSigmaPoints(self, sigma_points_pred, u_t, R_t):
        # state prediction
        weights = self.getWeights()
        state_pred = [[0] for _ in self.nDOF]
        for i in range(2 * self.nDOF + 1):
            state_pred = state_pred + weights[i] * sigma_points_pred[i]
            
        # update covariance matrix
        sigma_pred = np.zeros((self.nDOF, self.nDOF))
        for i in range(2 * self.nDOF + 1):
            diff = sigma_points_pred[i] - state_pred
            sigma_pred = sigma_pred + weights[i] * (diff @ diff.T)
        G_u_t = self.get_G_u_t(self.state_est, u_t)
        sigma_pred = sigma_pred + G_u_t @ R_t @ G_u_t.T
            
        return (state_pred, sigma_pred)
    
    def getPredictedMeasurements(self, sigma_points):
        Z_t_pred = []
        for sigma_pt in sigma_points:
            z_t = self.get_z_pred(sigma_pt)
            Z_t_pred = Z_t_pred + [z_t]
        return Z_t_pred
    
    def correctionStep(self, state_pred, sigma_pred, sigma_points_pred, z_t, Q_t):
        Z_t_pred = self.getPredictedMeasurements(sigma_points_pred)
        weights = self.getWeights()
        z_t_pred = [[0] for _ in self.nMeas]
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
            meas_diff = Z_t_pred[i] - z_t_pred
            sigma_x_z_t = sigma_x_z_t + weights[i] * (state_diff @ meas_diff.T)
        
        K_t = sigma_x_z_t @ np.linalg.inv(S_t)
        state_est = state_pred + K_t @ (z_t - z_t_pred)
        # TODO: move all wrapping to pi to a separate function
        state_est[THETA_INDEX, 0] = wrap_to_pi(state_est[THETA_INDEX, 0]) # clip theta
        sigma_est = sigma_pred - (K_t @ S_t @ K_t.T)
        
        self.state_est = state_est
        self.sigma_est = sigma_est

class UKFType(UKFBaseType):
    def applyMotionModel(sigma_points, u_t):
        # predict next state for each sigma point
        deltaSLeft = u_t[0, 0]
        deltaSRight = u_t[1, 0]
        
        deltaS = (deltaSRight + deltaSLeft) / 2
        deltaTheta = (deltaSRight - deltaSLeft) / EgoType.W
        
        sigma_points_pred = []
        for sigma_pt in sigma_points:
            x = sigma_pt[0, 0]
            y = sigma_pt[1, 0]
            theta = sigma_pt[2, 0]
            
            deltaX = deltaS * np.cos(theta + deltaTheta / 2)
            deltaY = deltaS * np.sin(theta + deltaTheta / 2)
            
            x_pred = x + deltaX
            y_pred = y + deltaY
            theta_pred = theta + deltaTheta # note that we do not want to clip this
            
            state_pred = np.matrix([[x_pred],
                                    [y_pred],
                                    [theta_pred]])
            sigma_points_pred = sigma_points_pred + [state_pred]
        return sigma_points_pred
        
    def localize(self, a_f, a_r, thetaDot, v_f, gps_x, gps_y):
        # control input
        u_t = np.matrix([[a_f], [a_r], [thetaDot]])
        R_t = np.diag([1**2, 1**2, 1**2]) # TODO: aaaa
    
        # prediction step using sigma points
        sigma_points = UKFType.getSigmaPoints(self.state_est, self.sigma_est, self.nDOF, self.scaling_factor)
        sigma_points_model = UKFType.applyMotionModel(sigma_points, u_t)
        (state_pred, sigma_pred) = self.regroupSigmaPoints(sigma_points_model, u_t, R_t)
        sigma_points_pred = UKFType.getSigmaPoints(state_pred, sigma_pred, self.nDOF, self.scaling_factor)
        
        # measurement
        z_t = np.matrix([[v_f], [gps_x], [gps_y]])
        Q_t = np.diag([1**2, 1**2, 1**2])
        
        # correction step
        outputs = self.correctionStep(state_pred, sigma_pred, sigma_points_pred, z_t, Q_t)
        
        return outputs
    
class UKFWithoutGPSType(UKFBaseType):
    def applyMotionModel(sigma_points, u_t):
        # predict next state for each sigma point
        # TODO: aaa
        pass
    
    def localize(self, a_f, a_r, thetaDot, v_f):
        # control input
        u_t = np.matrix([[a_f], [a_r], [thetaDot]])
        R_t = np.diag([1**2, 1**2, 1**2]) # TODO: aaaa
    
        # prediction step using sigma points
        sigma_points = UKFType.getSigmaPoints(self.state_est, self.sigma_est, self.nDOF, self.scaling_factor)
        sigma_points_model = UKFType.applyMotionModel(sigma_points, u_t)
        (state_pred, sigma_pred) = self.regroupSigmaPoints(sigma_points_model, u_t, R_t)
        sigma_points_pred = UKFType.getSigmaPoints(state_pred, sigma_pred, self.nDOF, self.scaling_factor)
        
        # measurement
        z_t = np.matrix([[v_f]])
        Q_t = np.diag([1**2])
        
        # correction step
        outputs = self.correctionStep(state_pred, sigma_pred, sigma_points_pred, z_t, Q_t)
        
        return outputs