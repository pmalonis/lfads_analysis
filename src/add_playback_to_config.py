import yaml
from glob import glob
import os

if __name__=='__main__':
    output_file = '../config.yml'

    raw_root = '/home/macleanlab/peter/lfads_analysis/data/raw/Playback-NN/split_condition/'
    params = ['param_kWj--O']
    lfads_root = '/home/macleanlab/peter/lfads_analysis/data/model_output/playback_first/param_kWJ--O/'
    server = 'macleanlab@205.208.22.226:'
    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    dsets = ['mk080729_M1m'] #os.listdir(raw_root)
    conds = os.listdir(raw_root + dsets[0])
    conds = [c.split('.')[0] for c in conds]
    datasets = cfg['datasets']
    for dset in dsets:
        for cond in conds:
            raw_path = raw_root + dset + '/' + cond + '.mat'
            info_path = lfads_root + 'single_%s/lfadsInput/inputInfo_%s.mat'%(cond,cond)
            param_dict = {}
            for param in params:
                for trial_type in ['valid', 'train']:
                    param_dict[trial_type] = server + lfads_root + 'single_' + cond + '/lfadsOutput/model_runs_active.h5_%s_posterior_sample_and_average'%trial_type
            dset_dict = {'inputInfo':info_path, 
                        'raw': server + raw_path, 
                        param:param_dict}
            dset_key = dset.split('_')[0] + '_' + cond
            datasets[dset_key] = dset_dict

    yaml.dump(cfg, open(output_file, 'w'))