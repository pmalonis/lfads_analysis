import pandas as pd
import h5py
import yaml
import matplotlib.pyplot as plt
import inspect
import sys
import os
import seaborn as sns
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from optimize_target_prediction import get_inputs_to_model

if __name__=='__main__':
    rand_controller_filename = snakemake.input[0]
    rand_rate_filename = snakemake.input[1]
    output = pd.read_csv(rand_controller_filename)
    rate_output = pd.read_csv(rand_rate_filename)
    
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = [(v['name'],k) for k,v in run_info.items()]

    output['Control Type'] = 'Initial'
    rate_output['Control Type'] = 'Corrective'
    plot_columns = ['reference', 'total_test_score', 'std_test_score', 'Control Type']
    plt.figure(figsize=(12,5))
    preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(output.columns)
    rate_preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(rate_output.columns)
    for i, (dset, file_root) in enumerate(datasets):
        dset_out = output.query('dataset==@dset')

        params = dset_out.query('reference=="hand"')
        params = dset_out.query('reference=="shoulder"')

        score = params['total_test_score']

        rate_dset_out = rate_output.query('dataset==@dset')

        rate_params = dset_out.query('reference=="hand"')

        rate_score = rate_params['total_test_score']

        plot_df = pd.DataFrame({'r^2':[score, rate_score],
                                'Predictor': ['Controller', 'Rate']})

        plt.subplot(1,len(datasets),i+1)
        sns.pointplot(x='Predictor', y='r^2', data=plot_df)
        plt.title(dset)
        plt.ylim([0, 0.5])
    
    plt.savefig(os.path.dirname(__file__) + 
                '/../../figures/final_figures/random_events.svg')