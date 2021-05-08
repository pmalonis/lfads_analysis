import numpy as np
from scipy import io
from convert_to_pandas import filter_kinematics, upsample_2x

if __name__=='__main__':
    speed_threshold = 1 #speed threhsold to count samples towards elimination cutoff
    trial_cutoff = 500 #trials with more than this many samples below threshold will be eliminated  
    raw_filename = "../data/raw/raju.mat"
    output_name = "../data/raw/raju_no_bad_trials.mat"
    data = io.loadmat(raw_filename)

    #getting trial cutoff
    T=data['cpl_st_trial_rew'][:,1] - data['cpl_st_trial_rew'][:,0]
    T = np.sort(T)
    total = np.zeros(T.shape[0])
    for i in range(T.shape[0]):
        total[i] = (T.shape[0]-i)*T[i]

    idx = np.argmax(total)
    trialCutoff = T[idx]

    trials = data['cpl_st_trial_rew']
    included = []
    speeds = []
    for i in range(data['cpl_st_trial_rew'].shape[0]):
        start = data['cpl_st_trial_rew'][i,0]
        stop = start + trialCutoff
        x_idx = (data['x'][:,0] >= start) & (data['x'][:,0] < stop)
        #y_idx = (data['y'][:,0] >= start) & (data['y'][:,0] < stop)
        t = upsample_2x(data['x'][x_idx,0])
        x_raw = data['x'][x_idx,1]
        y_raw = data['y'][x_idx,1]
        
        x_smooth, y_smooth = filter_kinematics(x_raw, y_raw)

        x = upsample_2x(x_smooth)
        y = upsample_2x(y_smooth)

        x_vel = np.gradient(x, t)
        y_vel = np.gradient(y, t)

        # filtering velocity, again
        x_vel, y_vel = filter_kinematics(x_vel, y_vel)
        speed = np.sqrt(x_vel**2 + y_vel**2)
        speeds.append(speed)
        if np.sum(speed < speed_threshold) < trial_cutoff:
            included.append(i)

    data['cpl_st_trial_rew'] = data['cpl_st_trial_rew'][included,:]
    io.savemat(output_name, data)