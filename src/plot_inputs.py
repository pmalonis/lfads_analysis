from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

if __name__ == '__main__':
    data_filename = "data/intermediate/rockstar.p"
    inputInfo_filename = "data/model_output/rockstar_valid.mat"
    valid_filename = "data/model_output/rockstar_valid.h5"
    input_info_file = "data/model_output/rockstar_inputInfo.mat"
    input_info = io.loadmat(input_info_file)

    #subtracting to convert to 0-based indexing
    train_inds = input_info['trainInds'][0] - 1
    valid_inds = input_info['validInds'][0] - 1

    df = pd.read_pickle(data_filename)
    dt = 0.010

    with h5py.File(valid_filename,'r') as h5_file:
        trial_len = h5_file['controller_outputs'].shape[1] * dt
        os.makedirs('param_xsgZ0x_input_timing_plots/valid/', exist_ok=True)
        for i in range(h5_file['controller_outputs'].shape[0]):
            #input = abs(h5_file['controller_outputs'][i,:,0]) +  abs(h5_file['controller_outputs'][i,:,1])
            input1 = h5_file['controller_outputs'][i,:,0]
            input2 = h5_file['controller_outputs'][i,:,1]
            targets = df.loc[valid_inds[i]].kinematic.loc[:trial_len].query('hit_target').index.values
            t = np.arange(0, trial_len, dt)
            try:
                fig = plt.figure()
                plt.plot(t, input1)
                plt.plot(t, input2)
                plt.vlines(targets, *fig.axes[0].get_ylim())
                plt.xlabel("Time (s)")
                plt.ylabel("Input value")
                plt.legend(["Input 1", "Input 2"])
                plt.savefig('param_xsgZ0x_input_timing_plots/valid/valid_%03d'%i)
                plt.close()
            except:
                import pdb;pdb.set_trace()