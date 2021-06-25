import numpy as np
import pandas as pd
import h5py
from scipy import io
import utils
import yaml
import os
import peak_trainvalidtest_split as ps
import matplotlib.pyplot as plt
import subsample_analysis as sa
from scipy.signal import savgol_filter, periodogram
from scipy.stats import kurtosis
import pickle
import timing_analysis as ta
import decode_lfads as dl
from plot_all_controller_metrics import metric_dict
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

absolute_min_heights = 0.3
relative_min_heights = 3
win_start = 0.0
win_stop = 0.3
peak_with_threshold = 0.9
figsize = (12,5)
n_splits = 5

run_info_path = os.path.join(os.path.dirname(__file__), '../lfads_file_locations.yml')
run_info = yaml.safe_load(open(run_info_path, 'r'))

if __name__=='__main__':
    metric_name = snakemake.wildcards.metric
    measure = metric_dict[metric_name](filename=snakemake.output[0])
    for dataset in run_info.keys():
        df = pd.read_pickle(os.path.dirname(__file__)+'/../data/intermediate/%s.p'%dataset)
        for param in run_info[dataset]['params'].keys():
            lfads_filename = os.path.dirname(__file__) + '/../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
            inputInfo_filename = os.path.dirname(__file__) + '/../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
            input_info = io.loadmat(inputInfo_filename)
            with h5py.File(lfads_filename, 'r') as h5file:
                co = h5file['controller_outputs'][:]
                dt = utils.get_dt(h5file, input_info)
                trial_len = utils.get_trial_len(h5file, input_info)

            prior = run_info[dataset]['params'][param]['param_values']['ar_prior_dist']
            kl_weight = run_info[dataset]['params'][param]['param_values']['kl_co_weight']
            measure.add_run(prior, kl_weight, dataset, param, df, co, trial_len, dt)
    
    measure.plot()
    measure.savefig()