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
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) +  '..')
import utils
from optimize_target_prediction import get_inputs_to_model
import optimize_target_prediction as opt
import model_evaluation as me
plt.rcParams['font.size'] = 20
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

config_path = os.path.join(os.path.dirname(__file__), '../../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

random_state = cfg['split_random_state']
r = cfg['significance_cv_repeats']
k = cfg['significance_cv_folds']

def cross_val_test_reference(output, dset_name, test_peak_df, co, trial_len, dt, df):
    '''Fit best model for each reference frame on test data and record results'''

    kf = RepeatedKFold(k, r, random_state=random_state)
    references = ['hand', 'shoulder']
    out_dict = {}
    scorer = make_scorer(opt.r_score_func)
    for reference in references:
        best_idx = output.query('fit_direction & ~use_rates & reference == @reference & dataset == @dset_name')['mean_test_r_score'].idxmax()
        model_row = output.loc[best_idx]
        preprocess_dict, model = me.get_row_params(model_row)
        X, y = get_inputs_to_model(test_peak_df, co, trial_len, 
                                                dt, df, **preprocess_dict)
        out_dict[reference] = cross_val_score(model, X, y, 
                                                scoring=scorer, 
                                                cv=kf)

    _, p = me.corrected_ttest(out_dict['hand']-out_dict['shoulder'], k, r)

    out_dict['p'] = p

    return out_dict

# def cross_val_test_reference(output, dset_name, test_peak_df, co, trial_len, dt, df):
#     '''Fit best model for each reference frame on test data and record results'''

#     kf = RepeatedKFold(k, r, random_state=random_state)
#     references = ['hand', 'shoulder']
#     scorer_names = ['x','y','r']
#     scorers = [make_scorer(opt.x_score_func), make_scorer(opt.y_score_func), make_scorer(opt.r_score_func)]
#     out_dict = {}
#     scorer = make_scorer(opt.var_weighted_score_func)
#     for reference in references:
#         best_idx = output.query('fit_direction & ~use_rates & reference == @reference & dataset == @dset_name')['mean_test_cosine_score'].idxmax()
#         model_row = output.loc[best_idx]
#         preprocess_dict, model = me.get_row_params(model_row)
#         X, y = get_inputs_to_model(test_peak_df, co, trial_len, 
#                                                 dt, df, **preprocess_dict)
#         out_dict[reference] = cross_val_score(model, X, y, 
#                                                 scoring=scorer, 
#                                                 cv=kf)
#         print(model)
    
#     _, p = me.corrected_ttest(out_dict['hand']-out_dict['shoulder'], k, r)

#     out_dict['p'] = p

#     return out_dict


if __name__=='__main__':
    plot_data_path = '../../data/model_output/hand_v_shoulder_correction_magnitude.p'
    if os.path.exists(plot_data_path):
        all_plot_dfs = pd.read_pickle(plot_data_path)
    else:
        output_filename = "../../data/peaks/params_search_corrections.csv"
            
        #output_filename = snakemake.input[0]
        #fb_output_filename = snakemake.input[1]

        output = pd.read_csv(output_filename)
        #fb_output = pd.read_csv(fb_output_filename)

        run_info = yaml.safe_load(open(os.path.dirname(__file__) + '../../lfads_file_locations.yml', 'r'))
        datasets = list(run_info.keys())
        output['Control Type'] = 'Initial'
        #fb_output['Control Type'] = 'Corrective'
        plot_score = 'final_held_out_score'
        plot_columns = ['reference', plot_score, 'std_test_score', 'Control Type', 'final_held_out_std']
        preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(output.columns)
        #fb_preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(fb_output.columns)

        colors = utils.contrasting_colors(**cfg['colors']['correction_decode'])
        all_cv_results = []
        for dataset in datasets:
            lfads_params = open('../../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read().strip()
            data_filename = '../../data/intermediate/' + dataset + '.p'
            lfads_filename = '../../data/model_output/' + \
                        '_'.join([dataset, lfads_params, 'all.h5'])
            inputInfo_filename = '../../data/model_output/' + \
                            '_'.join([dataset, 'inputInfo.mat'])
            peak_filename = '../../data/peaks/' + \
                        '_'.join([dataset, 'corrections_test.p'])

            df, co, trial_len, dt = utils.load_data(data_filename, lfads_filename, inputInfo_filename)
            peak_df = pd.read_pickle(peak_filename)
            dset_name = run_info[dataset]['name']
            cv_results = cross_val_test_reference(output, dset_name, 
                                                peak_df, co, trial_len, dt, df)
            print('%s hand performance: %f'%(dset_name, np.mean(cv_results['hand'])))
            print('%s shoulder performance: %f'%(dset_name, np.mean(cv_results['shoulder'])))
            print('%s p-value: %f'%(dset_name, cv_results['p']))
            all_cv_results.append(cv_results)

        plt.figure(figsize=(10,8))
        all_plot_dfs = []
        for i, (dataset, dset_cv_results) in enumerate(zip(datasets, all_cv_results)):
            dset_name = run_info[dataset]['name']
            dset_plot_df = pd.DataFrame({'r^2': np.concatenate([dset_cv_results['hand'], dset_cv_results['shoulder']]),
                                        'Monkey': [dset_name]*r*k*2,
                                        'Reference': ['Hand']*r*k + ['Shoulder']*r*k})
            all_plot_dfs.append(dset_plot_df)
            #plt.subplot(1, len(datasets), i+1)

        all_plot_dfs = pd.concat(all_plot_dfs)
        all_plot_dfs.to_pickle(plot_data_path)
    
    sns.pointplot(x='Reference', y='r^2', data=all_plot_dfs, hue='Monkey', linewidth=50, ci='sd')
    #sns.pointplot(x='Reference', y='r^2', data=plot_df.query('`Movement Type` == "Initial"'))

    #plt.errorbar([0,1], [hand_score.values[0], shoulder_score.values[0]], [hand_std.values[0], shoulder_std.values[0]], color=colors[0], linewidth=3)
    #plt.title("Monkey " + dset_name)
    #plt.ylim([-0.0,0.7])
    plt.xlim([-.5,1.5])
    _,ymax = plt.ylim()
    plt.ylim([-0.1, ymax])
    plt.title('Magnititude Decoder')
    plt.ylabel('Decoding Performance ($\mathregular{r^2}$)')

plt.savefig("../../figures/final_figures/numbered/6f.pdf")
    #plt.savefig(snakemake.output[0])