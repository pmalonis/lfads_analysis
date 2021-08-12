import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
from scipy import io
sys.path.insert(0, '..')
import decode_lfads as dl
import utils

if __name__=='__main__':
    n_splits = 5
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = run_info.keys()
    params = []
    for dataset in datasets:
        params.append(open('../../data/peaks/%s_selected_param_gini.txt'%(dataset)).read())

    plt.figure()
    for i, (dataset, param) in enumerate(zip(datasets, params)):
        data_filename = '../../data/intermediate/' + dataset + '.p'
        lfads_filename = '../../data/model_output/' + '_'.join([dataset, param, 'all.h5'])        
        inputInfo_filename = '../../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
        df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)

        with h5py.File(lfads_filename) as h5file:
            X_lfads = dl.get_lfads_predictor(h5file['factors'][:])

        X_smoothed = dl.get_smoothed_rates(df, trial_len, dt, 
                                        kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                                        used_inds=None)
        Y = dl.get_kinematics(df, trial_len, dt, 
                            kinematic_vars=['x', 'y', 'x_vel', 'y_vel'], 
                            used_inds=None)
        rs, _ = dl.get_rs(X_lfads, Y, n_splits, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                        use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs = [np.mean([rs[k][i] for k in rs.keys()]) for i in range(n_splits)]


        rs_smoothed, _ = dl.get_rs(X_smoothed, Y, n_splits, kinematic_vars = ['x', 'y', 'x_vel', 'y_vel'], 
                    use_reg=False, regularizer='ridge', alpha=1, random_state=None)

        mean_rs += [np.mean([rs_smoothed[k][i] for k in rs_smoothed.keys()]) 
                    for i in range(n_splits)]

        predictor = ['LFADS Factors']*n_splits + ['Firing Rates']*n_splits

        plot_df = pd.DataFrame.from_dict({'Predictor':predictor,'Variance Explained':mean_rs})
        plt.subplot(1, len(datasets), i+1)
        sns.pointplot(x='Predictor', y='Variance Explained', data=plot_df)
        plt.title(run_info[dataset]['name'])
        plt.savefig("../../figures/final_figures/%s_kinematic_decoding.png"%run_info[dataset]['name'])
