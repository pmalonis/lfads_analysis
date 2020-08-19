import numpy as np
from scipy import io
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score
from scipy.signal import find_peaks

win_size = 25 #number of samples of moving average window for kinematic data

def smooth(kinematics):
    win = np.ones(win_size)/float(win_size)
    kinematics = np.convolve(kinematics, win, 'valid')

    return kinematics

def get_curvature(x, y, t):
    """computes curvature of trajectory

    Parameters:
    dictionary of loaded data from mat file

    Output:
    k
    curvature at each sample in the kinematic trajectory
    """
    x_t = np.gradient(x, t)
    x_tt = np.gradient(x_t, t)
    y_t = np.gradient(y, t)
    y_tt = np.gradient(y_t, t)

    k = np.abs(x_tt*y_t - x_t*y_tt)/((x_t**2+y_t**2)**(3/2))

    return k

def get_speed(x, y, t):
    """computes curvature of trajectory
    Parameters:
    dictionary of loaded data from mat file

    Output:
    k
    curvature at each sample in the kinematic trajectory
    """
    x_t = np.gradient(x, t)
    x_tt = np.gradient(x_t, t)

    y_t = np.gradient(y, t)
    y_tt = np.gradient(y_t, t)

    v = np.sqrt(x_t**2 + y_t**2)

    return v

def get_power_law_deviation(x, y, t):
    """computes deviation of each sample from power law between
    speed and curvature (residual of regression of log speed to log curvature)"""
    log_speed = np.log(get_speed(x, y, t))
    log_curvature = np.log(get_curvature(x, y, t))

    lr = linregress(log_curvature, log_speed)

    return np.abs((lr.slope * log_curvature + lr.intercept) - log_speed)
    

def get_normal_accel(x, y, t):
    """
    computes normal acceleration at each time point
    """
    x_t = np.gradient(x, t)
    y_t = np.gradient(y, t)

    x_accel = np.gradient(x_t, t)
    y_accel = np.gradient(y_t, t)

    speed = get_speed(x, y, t)
    x_tang_velocity = x_t/speed
    y_tang_velocity = y_t/speed
    normal_accel = np.sqrt(np.gradient(x_tang_velocity, t)**2 + np.gradient(y_tang_velocity, t)**2)

    return normal_accel

def get_non_target_peaks(df, ):

    return 

def roc_event(a, b, thresh):
    

    return 
    

if __name__=='__main__':
    filename = '../data/rs1050211_clean_spikes_SNRgt4.mat'

    mat_data = io.loadmat(filename)

    n_trials = mat_data['endmv'].shape[0]
    rvalues = []
    slopes = []
    starts = []
    stops = []
    for i in range(n_trials):
        start = mat_data['endmv'][i,0]
        stop = mat_data['st_trial'][mat_data['st_trial'] > start][0]

        if stop-start < 0.05:
            continue

        k = curvature(mat_data, start, stop)
        v = speed(mat_data, start, stop)

        idx = np.logical_and(np.log(k)>-20, np.log(k)<20)
        lr = linregress(np.log(k[idx]), np.log(v[idx]))
        starts.append(start)
        stops.append(stop)
        rvalues.append(lr.rvalue)
        slopes.append(lr.slope)
