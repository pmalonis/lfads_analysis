from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from utils import git_savefig, get_indices
from matplotlib.backends.backend_pdf import PdfPages
import subprocess as sp

# %%
if __name__ == '__main__':
    data_filename = snakemake.input[0]
    lfads_filename = snakemake.input[1]
    input_info_file = snakemake.input[2]

    dataset = snakemake.wildcards.dataset
    trial_type = snakemake.wildcards.trial_type
    param = snakemake.wildcards.param

    out_directory = 'figures/input_timing_plots/'
    os.makedirs(out_directory, exist_ok=True)

    input_info = io.loadmat(input_info_file)

    #subtracting to convert to 0-based indexing
    used_inds = get_indices(input_info, snakemake.wildcards.trial_type)

    df = pd.read_pickle(data_filename)
    trial_len_ms = input_info['seq_timeVector'][-1][-1]
    trial_len = trial_len_ms/1000

    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    with PdfPages(snakemake.output[0], metadata={'commit':commit}) as pdf:
        with h5py.File(lfads_filename,'r') as h5_file:
            dt = np.round(trial_len_ms/h5_file['controller_outputs'].shape[1])/1000
            trial_len = np.floor(trial_len/dt) * dt
            for i in range(h5_file['controller_outputs'].shape[0]):
                input1 = h5_file['controller_outputs'][i,:,0]
                input2 = h5_file['controller_outputs'][i,:,1]
                targets = df.loc[used_inds[i]].kinematic.loc[:trial_len].query('hit_target').index.values
                t = np.arange(0, trial_len, dt)
                fig = plt.figure()
                plt.plot(t, input1)
                plt.plot(t, input2)
                plt.vlines(targets, *fig.axes[0].get_ylim())
                plt.ylim([-1.5, 1.5])
                plt.xlabel("Time (s)")
                plt.ylabel("Input value")
                plt.legend(["Input 1", "Input 2"])
                plt.title("%s trial %03d"%(param, i))
                pdf.savefig(fig)
                plt.close()