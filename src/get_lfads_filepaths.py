import yaml
from parse import parse
import subprocess as sp
import os
from os.path import dirname, basename
import pandas as pd
import numpy as np
from io import StringIO
import re

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
#cfg['lfads_file_server'] = '205.208.22.225'
#cfg['username'] = 'macleanlab'
#cfg['lfads_dir_path'] = '/home/macleanlab/peter/lfads_analysis/data/model_output/'
if __name__=='__main__':
    output_filename = '../lfads_file_locations.yml'
    common_param_filename = '../lfads_common_parameters.yml'
    file_pattern = 'gaussian_kl_sweep/*/single*k*/*/model_runs_*.h5*_posterior_sample_and_average'
    file_pattern = cfg['lfads_dir_path'] + file_pattern
    file_pattern += ' ' + cfg['lfads_dir_path'] + 'laplace_kl_sweep/*/single_*k*/*/model_runs_*.h5*_posterior_sample_and_average'
    file_pattern += ' ' + cfg['lfads_dir_path'] + 'raju_wide_kl_range/*/*/*/model_runs_*.h5*_posterior_sample_and_average'
    file_pattern += ' ' + cfg['lfads_dir_path'] + 'wide_kl_range/*/single_rockstar/*/model_runs_*.h5*_posterior_sample_and_average'
    file_pattern += ' ' + cfg['lfads_dir_path'] + 'wide_kl_range/*/single_mack/*/model_runs_*.h5*_posterior_sample_and_average'
    files = sp.check_output(['ssh', cfg['username'] + "@" + cfg['lfads_file_server'], 'ls', file_pattern]).decode().split()
    hp_files=[dirname(f)+ '/hyperparameters-0.txt'
              for f in files if '.h5_train_posterior_sample_and_average' in f]
    hp_str = sp.check_output(['ssh', cfg['username'] + "@" + cfg['lfads_file_server'], 'cat'] + hp_files).decode()
    temp_delimiter = '|'
    hp_str_list = hp_str.replace('}{','}%s{'%temp_delimiter).split(temp_delimiter)
    hp_dicts = [yaml.safe_load(s) for s in hp_str_list]

    fitlog_files=[dirname(f) + '/fitlog.csv'
                    for f in files if '.h5_train_posterior_sample_and_average' in f]
    fitlog_str = sp.check_output(['ssh', cfg['username'] + "@" + cfg['lfads_file_server'], 'tail'] + fitlog_files).decode()
    fitlog_str_list = re.sub(r'==> .*fitlog.csv', '', fitlog_str).split('<==')
    
    #getting error from fitlog
    fit_dicts = []
    for fitlog in fitlog_str_list[1:]:
        table = pd.read_csv(StringIO(fitlog))
        total = table.iloc[-1,5:7]
        recon = table.iloc[-1,8:10]
        kl = table.iloc[-1,11:13]
        fit_dict = {}
        fit_dict['total_train'] = float(total[0])
        fit_dict['total_valid'] = float(total[1])
        fit_dict['recon_train'] = float(recon[0])
        fit_dict['recon_valid'] = float(recon[1])
        fit_dict['kl_train'] = float(kl[0])
        fit_dict['kl_valid'] = float(kl[1])
        fit_dicts.append(fit_dict)

    #parsing hyperparameter bools and removing non param keys
    for i in range(len(hp_dicts)):
        hp_dicts[i] = {k:(True if v=='true' else v) for k,v in hp_dicts[i].items()}
        hp_dicts[i] = {k:(False if v=='false' else v) for k,v in hp_dicts[i].items()}
        hp_dicts[i].pop('data_dir')
        hp_dicts[i].pop('lfads_save_dir')
        hp_dicts[i].pop('dataset_names')
        hp_dicts[i].pop('dataset_dims')
        hp_dicts[i].pop('num_steps')
        hp_dicts[i].pop('num_steps_for_gen_ic')

    #separating out common paramaters
    all_keys = list(set(k for hp_dict in hp_dicts for k in list(hp_dict.keys()) ))
    common_params = {}
    for k in all_keys:
        if k in hp_dicts[0].keys() and np.all([k in hp_dict.keys() and hp_dicts[0][k]==hp_dict[k] for hp_dict in hp_dicts[1:]]):
            common_params[k] = hp_dicts[0][k]
            for hp_dict in hp_dicts:
                hp_dict.pop(k)

    form = '{run}/param_{lfads_param}/{}/lfadsOutput/model_runs_{dataset}.h5_{subset}_posterior_sample_and_average'
    d = [parse(form, f).named for f in files]
    for x in d:
        x['run'] = basename(x['run'])

    df = pd.DataFrame(d)
    df['param'] = df['run'] + '-' + df['lfads_param']
    #df.drop(['run', 'lfads_param'], inplace=True, axis=1)
    df['path'] = ['%s@%s:%s'%(cfg['username'], cfg['lfads_file_server'], f) for f in files]
    # creates nested dictionary for yaml
    out_dict = df.groupby('dataset').apply(lambda x:x.groupby('param').apply(lambda x:x.set_index(['subset']).to_dict()['path']).to_dict()).to_dict()

    #adding params level for consistency with previous format
    for dataset in out_dict.keys():
        out_dict[dataset] = {'params':out_dict[dataset]}

    # adding input info and getting
    for dataset in out_dict.keys():
        for param in out_dict[dataset]['params'].keys():
            file_root = dirname(dirname(out_dict[dataset]['params'][param]['train']))
            out_dict[dataset]['params'][param]['inputInfo'] = file_root + '/lfadsInput/inputInfo_%s.mat'%dataset
            
    #adding hyperparameter values
    for hp_dict, fit_dict, (dataset, param) in zip(hp_dicts, fit_dicts, df.query('subset=="train"')[['dataset','param']].values):
        out_dict[dataset]['params'][param]['param_values'] = hp_dict
        out_dict[dataset]['params'][param]['fit'] = fit_dict

    #ad hoc fix for mack
    if 'mk08011M1m' in out_dict.keys():
        out_dict['mack']['params'].update(out_dict['mk08011M1m']['params'])
        out_dict.pop('mk08011M1m')

    #adding raw file paths
    for dataset in cfg['datasets'].keys():
        if dataset in out_dict.keys():
            out_dict[dataset].update(cfg['datasets'][dataset])

    #removing datasets that do not have raw path, and reordering
    out_dict = {dataset:out_dict[dataset] for dataset in cfg['datasets'].keys() if dataset in out_dict.keys()}

    #changing dataset and parameter names for easier parsing
    out_dict = {k.replace('_','-'):v for k,v in out_dict.items()}

    for dataset in out_dict.keys():
        out_dict[dataset]['params'] = {k.replace('_','-'):v for k,v in out_dict[dataset]['params'].items()}

    with open(output_filename, 'w') as out_file:
        yaml.safe_dump(out_dict, out_file, sort_keys=False)

    # for k in common_params.keys():
    #     if k[0] == '_':
    #         common_params[k[1:]] = common_params.pop(k)

    with open(common_param_filename, 'w') as common_param_file:
        yaml.safe_dump(common_params, common_param_file)