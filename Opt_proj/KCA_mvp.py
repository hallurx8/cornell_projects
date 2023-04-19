import numpy as np
from pykalman import KalmanFilter

def fitKCA(t,z,q,fwd=0):
    '''
    Inputs:
        t: iterable with time indices
        z: iterable with measurements
        q: scalar that multiplies the seed states covariance
        fwd: number of steps to forecast (default to 0)
    Outputs:
        x[0]: smoothed state means of position velocity and acceleration
        x[1]: smoothed state covar of position velocity and acceleration
    '''
    #monthly timestep
    h = (t[-1] - t[0])/t.shape[0]
    
    A = np.array([[1, h, 0.5*h**2],
                 [0, 1, h],
                 [0, 0, 1]])
    Q = q*np.eye(A.shape[0])

    #set up Kalman filter
    kf = KalmanFilter(transition_matrices=A, transition_covariance=Q)

    #EM for parameter esimtates
    kf = kf.em(z)

    #smooth
    x_mean, x_cov = kf.smooth(z)

    #forecast
    for fwd_ in range(fwd):
        x_mean_, x_cov_ = kf.filter_update(filtered_state_mean=x_mean[-1],
                                         filtered_state_covariance=x_cov[-1])
        x_mean = np.append(x_mean, x_mean_.reshape(1,-1), axis=0)
        x_cov_ = np.expand_dims(x_cov_, axis=0)
        x_cov = np.append(x_cov, x_cov_, axis=0)

    #std series
    x_std = (x_cov[:,0,0]**0.5).reshape(-1,1)
    for i in range(1, x_cov.shape[1]):
        x_std_ = x_cov[:,i,i]**0.5
        x_std = np.append(x_std, x_std_.reshape(-1,1), axis=1)
        
    return x_mean, x_std, x_cov