import yaml
from glob import glob1
import os

if __name__=='__main__':
    output_file = '../config.yml'
    raw_root = '/home/macleanlab/peter/lfads_analysis/data/raw/'
    lfads_root = '/home/macleanlab/peter/lfads_analysis/data/model_output/subsampled_rockstar/'
    params = [p[6:] for p in glob1(lfads_root, 'param_*')]
    server = 'macleanlab@205.208.22.226:'
    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))
    dsets = ['rockstar', ] #os.listdir(raw_root)
    #conds = os.listdir(raw_root + dsets[0])
    conds = ['300_trial', '400_trial', '500_trial', '600_trial']#active']#['dual_pb', 'vis_pb', 'prop_pb'] #[c.split('.')[0] for c in conds]
    datasets = cfg['datasets']
    for dset in dsets:
        for cond in conds:
            raw_path = raw_root + '/' + cond + '_' + dset + '.mat'
            info_path = server + lfads_root + 'param_%s/all/lfadsInput/inputInfo_%s.mat'%(params[0], cond)
            param_dict = {}
            for param in params:
                param_dict[param] = {}
                for trial_type in ['valid', 'train']:
                    model_path = server + lfads_root + 'param_' + param + '/all' + '/lfadsOutput/model_runs_%s_%s.h5_%s_posterior_sample_and_average'%(cond,dset,trial_type)
                    param_dict[param][trial_type] = model_path
            dset_dict = {'inputInfo':info_path, 
                        'raw': server + raw_path, 
                        'params':param_dict}
            dset_key = dset.split('_')[0] + '_' + cond
            datasets[dset_key] = dset_dict

    yaml.dump(cfg, open(output_file, 'w'))
