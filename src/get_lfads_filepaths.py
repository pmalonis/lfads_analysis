import yaml
from parse import parse
import subprocess as sp
import os
from os.path import dirname
import pandas as pd

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

if __name__=='__main__':
    output_filename = '../lfads_file_locations.yml'

    file_pattern = '*/*/*/*/model_runs_*.h5*_posterior_sample_and_average'
    file_pattern = cfg['lfads_dir_path'] + file_pattern
    files = sp.check_output(['ssh', cfg['username'] + "@" + cfg['lfads_file_server'], 'ls', file_pattern]).split()
    
    form = cfg['lfads_dir_path'] + '{run}/param_{lfads_param}/{}/lfadsOutput/model_runs_{dataset}.h5_{subset}_posterior_sample_and_average'
    files = [f.decode() for f in files]
    d = [parse(form, f).named for f in files] 
    df = pd.DataFrame(d)
    df['run'] = df['run'].str.replace('_','-')#for easier reading by snakemake
    df['param'] = df['run'] + '-' + df['lfads_param']
    df.drop(['run', 'lfads_param'], inplace=True, axis=1)
    df['path'] = ['%s@%s:%s'%(cfg['username'],cfg['lfads_file_server'], f) for f in files]
    # creates nested dictionary for yaml
    out_dict = df.groupby('dataset').apply(lambda x:x.groupby('param').apply(lambda x:x.set_index(['subset']).to_dict()['path']).to_dict()).to_dict()

    #ad hoc fix for mack
    if 'mk08011M1m' in out_dict.keys():
        out_dict['mack'] = out_dict['mk08011M1m']
        out_dict.pop('mk08011M1m')

    #adding params level for consistency with previous format
    for dataset in out_dict.keys():
        out_dict[dataset] = {'params':out_dict[dataset]}
        
    #adding raw file paths
    for dataset in cfg['raw_datasets'].keys():
        out_dict[dataset]['raw'] = cfg['raw_datasets'][dataset]

    # adding input info
    for dataset in out_dict.keys():
        for param in out_dict[dataset]['params'].keys():
            file_root = dirname(dirname(out_dict[dataset]['params'][param]['train']))
            out_dict[dataset]['params'][param]['inputInfo'] = file_root + '/inputInfo_%s.mat'%dataset

    with open(output_filename, 'w') as out_file:
        yaml.safe_dump(out_dict, out_file)