import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from scipy import io
sys.path.insert(0, '.')
import utils
from utils import git_savefig, get_indices
from matplotlib.backends.backend_pdf import PdfPages
import subprocess as sp

def smooth_trials(df, trial_len, used_inds=None, downsample_factor=10):    
    if used_inds is None:
        assert(peak_df.index[-1][0] + 1 == co.shape[0])
        used_inds = range(co.shape[0])

    win = 10 #smoothed_dt/spike_dt
    midpoint_idx = int((win-1)/2)
    nneurons = sum('neural' in c for c in df.columns)
    recording_dt = 0.001
    recording_dt *= downsample_factor
    all_smoothed = np.zeros((len(used_inds), int(trial_len/recording_dt), nneurons))
    for i in [0]:#used_inds:
        smoothed = df.loc[i].neural.rolling(window=300, min_periods=1, win_type='gaussian', center=True).mean(std=50)
        smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
        smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
        all_smoothed[i,:,:] = smoothed

    return all_smoothed

if __name__=='__main__':
    
    data_filename = '../data/intermediate/rockstar.p'#snakemake.input[0]
    lfads_filename = '../data/model_output/rockstar_8QTVEk_all.h5' #snakemake.input[1]
    input_info_file = '../data/model_output/rockstar_inputInfo.mat' #snakemake.input[2]

    #dataset = snakemake.wildcards.dataset
    #trial_type = snakemake.wildcards.trial_type
    #param = snakemake.wildcards.param

    # out_directory = 'figures/input_timing_plots/'
    # os.makedirs(out_directory, exist_ok=True)

    input_info = io.loadmat(input_info_file)

    #subtracting to convert to 0-based indexing
    trial_type = 'all'
    #used_inds = get_indices(input_info, snakemake.wildcards.trial_type)
    used_inds = utils.get_indices(input_info, trial_type)

    df = pd.read_pickle(data_filename)
    trial_len_ms = input_info['seq_timeVector'][-1][-1]
    trial_len = trial_len_ms/1000

    all_smoothed = smooth_trials(df, trial_len, used_inds)
    pca = PCA(n_components=2)
    pca.fit(np.vstack(all_smoothed))

    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    #with PdfPages(snakemake.output[0], metadata={'commit':commit}) as pdf:
    with h5py.File(lfads_filename,'r') as h5_file:
        dt = np.round(trial_len_ms/h5_file['controller_outputs'].shape[1])/1000
        trial_len = np.floor(trial_len/dt) * dt
        for i in [0]:#range(h5_file['controller_outputs'].shape[0]):
            input1 = h5_file['controller_outputs'][i,:,0]
            input2 = h5_file['controller_outputs'][i,:,1]
            targets = df.loc[used_inds[i]].kinematic.loc[:trial_len].query('hit_target').index.values
            t = np.arange(0, trial_len, dt)
            fig = plt.figure()
            plt.plot(t, input1)
            plt.plot(t, input2)
            
            pc = pca.transform(all_smoothed[i])
        
            plt.plot(t, pc[:,0])
            plt.plot(t, pc[:,1])

            plt.vlines(targets, *fig.axes[0].get_ylim())
            plt.ylim([-1.5, 1.5])
            plt.xlabel("Time (s)")
            plt.ylabel("Input value")
            plt.legend(["Input 1", "Input 2", "Spike PC 1", "Spike PC 2"])
            #plt.title("%s trial %03d"%(param, i))
                #pdf.savefig(fig)
      #          plt.close()

    plt.show()