import numpy as np
import pandas as pd
from scipy import io
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import inspect
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) +  '/..')
import utils
from optimize_target_prediction import get_inputs_to_model
plt.rcParams['font.size'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

random_state = 1748
estimator_dict = {'SVR': MultiOutputRegressor(SVR()), 
                  'Random Forest': RandomForestRegressor(random_state=random_state)}

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def test_model(model_row, train_peak_df, test_peak_df, input_info, df):
    '''Fit best model for each reference frame on test data and record results'''

    lfads_params = model_row['lfads_params'].values[0]
    lfads_filename = os.path.dirname(__file__) + '/../../data/model_output/' + \
                        '_'.join([file_root, lfads_params, 'all.h5'])
    with h5py.File(lfads_filename, 'r+') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    model_args = [c for c in model_row.columns if c[:6]=='param_' and model_row[c].notnull().values[0]]
    preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(model_row.columns)
    preprocess_dict = model_row[preprocess_args].iloc[0].to_dict()
    if 'win_lim' in preprocess_dict.keys():
        preprocess_dict['win_lim'] = eval(preprocess_dict['win_lim'])

    model_dict = model_row[model_args].iloc[0].to_dict()
    model_dict = {k[6:]:v for k,v in model_dict.items()}
    model = estimator_dict[model_row['estimator'].values[0]]
    if isinstance(model, MultiOutputRegressor):
        model.estimator.set_params(**model_dict)
    else:
        model.set_params(**model_dict)

    X_train, y_train = get_inputs_to_model(train_peak_df, co, trial_len, dt, df, **preprocess_dict)
    X_test, y_test = get_inputs_to_model(train_peak_df, co, trial_len, dt, df, **preprocess_dict)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return score

if __name__=='__main__':
    output_filename = "../../data/peaks/controller_best_models_targets-not-one.csv"
    fb_output_filename = "../../data/peaks/controller_best_models_corrections.csv"
        
    #output_filename = snakemake.input[0]
    #fb_output_filename = snakemake.input[1]

    output = pd.read_csv(output_filename)
    fb_output = pd.read_csv(fb_output_filename)

    run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../../lfads_file_locations.yml', 'r'))
    datasets = [(v['name'],k) for k,v in run_info.items()]
    print(datasets)
    output['Control Type'] = 'Initial'
    fb_output['Control Type'] = 'Corrective'
    plot_score = 'final_held_out_score'
    plot_columns = ['reference', plot_score, 'std_test_score', 'Control Type', 'final_held_out_std']
    preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(output.columns)
    fb_preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(fb_output.columns)
    plt.figure(figsize=(20,6))

    colors = utils.contrasting_colors(**cfg['colors']['correction_decode'])
    for i, (dset, file_root) in enumerate(datasets):
        dset_out = output.query('dataset==@dset')

        hand_params = dset_out.query('reference=="hand"')
        shoulder_params = dset_out.query('reference=="shoulder"')

        hand_score = hand_params[plot_score]
        shoulder_score = shoulder_params[plot_score]
        hand_std = hand_params['final_held_out_std']
        shoulder_std = shoulder_params['final_held_out_std']

        fb_dset_out = fb_output.query('dataset==@dset')

        fb_hand_params = fb_dset_out.query('reference=="hand"')
        fb_shoulder_params = fb_dset_out.query('reference=="shoulder"')

        fb_hand_score = fb_hand_params[plot_score]
        fb_shoulder_score = fb_shoulder_params[plot_score]
        fb_hand_std = fb_hand_params['final_held_out_std']
        fb_shoulder_std = fb_shoulder_params['final_held_out_std']

        plot_df = pd.DataFrame({'r^2':[hand_score, shoulder_score, fb_hand_score, fb_shoulder_score],
                                'Movement Type': ['Initial', 'Initial', 'Corrective', 'Corrective'],
                                'Reference': ['hand', 'shoulder', 'hand', 'shoulder']})

        plt.subplot(1, len(datasets), i+1)
        sns.pointplot(x='Reference', y='r^2', hue='Movement Type', data=plot_df, palette=colors, linewidth=3)
        #sns.pointplot(x='Reference', y='r^2', data=plot_df.query('`Movement Type` == "Initial"'))

        plt.errorbar([0,1], [hand_score.values[0], shoulder_score.values[0]], [hand_std.values[0], shoulder_std.values[0]], color=colors[0], linewidth=3)
        plt.errorbar([0,1], [fb_hand_score.values[0], fb_shoulder_score.values[0]], [fb_hand_std.values[0], fb_shoulder_std.values[0]], color=colors[1], linewidth=3)
        plt.title("Monkey " + dset)
        #plt.ylim([-0.0,0.7])
        #plt.xlim([-.5,1.5])
        plt.ylabel('$\mathregular{r^2}$')

    #plt.savefig("../../figures/final_figures/hand_v_shoulder_corrective.png")
    plt.savefig("../../figures/final_figures/hand_v_shoulder_corrective.svg")
    plt.savefig('../../figures/final_figures/numbered/6d.svg')
    #plt.savefig(snakemake.output[0])