from scipy.stats import norm
import h5py
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import glob
sys.path.insert(0, '..')
import utils
import segment_submovements as ss
from importlib import reload
reload(ss)
reload(utils)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = True
plt.rcParams['font.size'] = 18

# removing old pngs
for png in glob.glob('../figures/speed_with_corrections/*.png'):
    os.remove(png)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
datasets = list(run_info.keys())
params = []
for dataset in run_info.keys():
    params.append(open('../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read())

for dset_idx, (dataset, param) in enumerate(zip(datasets, params)):
    
    data_filename = '../data/intermediate/' + dataset + '.p'
    lfads_filename = '../data/model_output/' + '_'.join([dataset, param, 'all.h5'])
    inputInfo_filename = '../data/model_output/' + '_'.join([dataset, 'inputInfo.mat'])
    
    df = data_filename = pd.read_pickle(data_filename)
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename, 'r') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    #fm = pd.read_pickle('../data/peaks/%s_new-firstmove_all.p'%(dataset))
    fm = pd.read_pickle('../data/peaks/%s_first-accel_all.p'%(dataset))
    c = pd.read_pickle('../data/peaks/%s_new-corrections_all.p'%(dataset))

    n_trials = df.index[-1][0] + 1
    #for i in range(n_trials):
    for i in range(50):
        firstmoves = fm.loc[i] if i in fm.index.get_level_values('trial') else pd.DataFrame([])
        corrections = c.loc[i] if i in c.index.get_level_values('trial') else pd.DataFrame([])
        ss.plot_trial(df.loc[i], co[i,:,:], 0.01, firstmoves, corrections)
        plt.savefig('../figures/speed_with_corrections/%s_%03d.png'%(dataset,i))
        if i%10 == 0:
            print('Plotted speed for %s trial %d'%(dataset, i))
        plt.close()

    print('Creating PDF')
    os.system('convert ../figures/speed_with_corrections/%s_*.png ../figures/speed_with_corrections/%s_speed.pdf'%(dataset,dataset))
    for png in glob.glob('../figures/speed_with_corrections/*.png'):
        os.remove(png)