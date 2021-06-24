import numpy as np
from scipy import io
import h5py
import pandas as pd
from convert_to_pandas import filter_kinematics, upsample_2x

if __name__=='__main__':
    speed_threshold = 1 #speed threhsold to count samples towards elimination cutoff
    trial_cutoff = 500 #trials with more than this many samples below threshold will be eliminated  
    raw_filename = "../data/raw/raju-M1.mat"
    pandas_filename = "../data/intermediate/raju.p"
    lfads_filename = "../data/model_output/raju_final-fixed-2OLS24_all.h5"
    output_name = "raju-M1-no-bad-trials"
    lfads_output = "../data/model_output/raju-M1-no-bad-trials_final-fixed-2OLS24_all.h5"
    data = io.loadmat(raw_filename)

    fr_threshold = 150
    binsize = 0.1

    #getting trial cutoff
    T = data['cpl_st_trial_rew'][:,1] - data['cpl_st_trial_rew'][:,0]
    T = np.sort(T)
    total = np.zeros(T.shape[0])
    for i in range(T.shape[0]):
        total[i] = (T.shape[0]-i)*T[i]

    idx = np.argmax(total)
    trialCutoff = T[idx]

    trials = data['cpl_st_trial_rew']
    mat_included = []
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
        #speeds.append(speed)
        if np.sum(speed < speed_threshold) < trial_cutoff:
            mat_included.append(i)

    data['cpl_st_trial_rew'] = data['cpl_st_trial_rew'][mat_included,:]
    neurons = [k for k in data.keys() if 'Chan' in k]
    T = np.max([data[n][-1,:] for n in neurons])
    max_rates = np.array([np.histogram(data[n],np.arange(0,T,binsize))[0].max()*(1/binsize) 
                            for n in neurons])
    thresh_exclude = [n for i,n in enumerate(neurons) if max_rates[i] > fr_threshold]
    ad_hoc_exclude = ['Chan002a', 'Chan051a']
    exclude = set(thresh_exclude + ad_hoc_exclude)
    for neuron in list(exclude):
        if neuron in data.keys():
            data.pop(neuron)
        
    io.savemat("../data/raw/%s.mat"%output_name, data)

    # del data

    # with h5py.File(lfads_filename, 'r') as h5file:
    #     co = h5file['controller_outputs'][:]

    # df_included = []
    # df = pd.read_pickle(pandas_filename)
    # assert(df.index[-1][0]+1 == co.shape[0])
    # for i in range(co.shape[0]):
    #     x_vel, y_vel = df.loc[i].kinematic.loc[:trialCutoff][['x_vel','y_vel']].values.T
    #     speed = np.sqrt(x_vel**2 + y_vel**2)
    #     if np.sum(speed < speed_threshold) < trial_cutoff:
    #         df_included.append(i)
        
    # df = df.loc[df_included]
    # trials = np.concatenate([[i]*df.loc[idx].shape[0] for i,idx in 
    #                          zip(range(301), list(set(df.index.get_level_values('trial'))))])
    # times = df.index.get_level_values('time')
    # df.index = pd.MultiIndex.from_tuples(zip(trials,times),names=['trial','time'])  
    # df.to_pickle("../data/intermediate/%s.p"%output_name)
    # with h5py.File(lfads_output, 'w') as outh5file:
    #     outh5file['controller_outputs'] = co[df_included]