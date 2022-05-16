import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t_start = 0.05
t_end = .25

def get_first_targets(_df):
    trial_df = _df.loc[_df.index[0][0]]
    targets = trial_df.loc[trial_df.kinematic.query('hit_target').index]
    if targets.index[1] - targets.index[0] < t_end:
        return None

    theta = np.arctan2(targets.iloc[1:].kinematic['y_vel'], targets.iloc[1:].kinematic['x_vel'])
    npthetas = np.append(theta.to_numpy(), np.nan)
    targets['theta'] = npthetas

    targets = targets.iloc[0:1]
    t_target = targets.index[0]

    sum_spikes = pd.DataFrame(trial_df.loc[t_target + t_start:t_target + t_end].neural.sum(0)).T
    sum_spikes.set_index(targets.index, inplace=True)

    out = pd.concat([targets.theta, sum_spikes], axis=1,
                          keys=['theta','sum_spikes'])

    return out

def get_second_targets(_df):
    trial_df = _df.loc[_df.index[0][0]]
    targets = trial_df.loc[trial_df.kinematic.query('hit_target').index]
    if targets.index[2] - targets.index[1] < t_end:
        return None

    theta = np.arctan2(targets.iloc[1:].kinematic['y_vel'], targets.iloc[1:].kinematic['x_vel'])
    npthetas = np.append(theta.to_numpy(), np.nan)
    targets['theta'] = npthetas

    targets = targets.iloc[1:2]
    t_target = targets.index[0]

    sum_spikes = pd.DataFrame(trial_df.loc[t_target + t_start:t_target + t_end].neural.sum(0)).T
    sum_spikes.set_index(targets.index, inplace=True)

    out = pd.concat([targets.theta, sum_spikes], axis=1,
                          keys=['theta','sum_spikes'])
    
    return out

if __name__=='__main__':
    theta = pickle.load(open('fit_direction_Transient_posture_serial_3targ.p','rb'))['theta']
    rates = pickle.load(open('fit_direction_Transient_posture_serial_3targ.p','rb'))['rates'].numpy()
    bins = np.arange(0, 2*np.pi+np.pi/3, np.pi/3)

    first_rates =  pd.DataFrame(rates[:,:,:20].mean(2))
    second_rates =  pd.DataFrame(rates[:,:,40:].mean(2))
    first_theta = pd.DataFrame(theta[:,0])
    second_theta = pd.DataFrame(theta[:,2])
    first_targets = pd.concat([first_theta,first_rates], axis=1, keys=['theta','rates'])
    second_targets = pd.concat([second_theta,second_rates], axis=1, keys=['theta','rates'])
    first_targets['theta_bin'] = pd.cut(first_targets.theta.to_numpy().flatten(), bins=bins)
    second_targets['theta_bin'] = pd.cut(second_targets.theta.to_numpy().flatten(), bins=bins)
    n_neurons = rates.shape[1]
    coefs = np.zeros(n_neurons)
    coefs_pos = []
    k = 0
    for i in range(n_neurons):
        first_tuning = first_targets.groupby('theta_bin').mean().rates.iloc[:,i]/(t_end-t_start)
        second_tuning = second_targets.groupby('theta_bin').mean().rates.iloc[:,i]/(t_end-t_start)
        coefs[i] = np.corrcoef(first_tuning, second_tuning)[0,1]
        bin_centers = bins[:-1] + (bins[1]-bins[0])/2
        if second_tuning.mean() > 0.0001:
            coefs_pos.append(coefs[i])
            if k < 10:
                plt.figure()
                plt.plot(bin_centers, first_tuning)
                plt.plot(bin_centers, second_tuning)
                plt.legend(['First Target', 'Second Target'])
                k += 1

    plt.show()



