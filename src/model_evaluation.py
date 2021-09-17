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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools
import os
import sys
from ast import literal_eval
from sklearn.utils import resample
sys.path.insert(0, '..')
import utils
from optimize_target_prediction import get_inputs_to_model

random_state = 1748
estimator_dict = {'SVR': MultiOutputRegressor(SVR()),
                  'Random Forest': RandomForestRegressor(random_state=random_state)}

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def get_row_params(model_row):

    model_args = [c for c in model_row.index if c[:6]=='param_' and 
                    not (isinstance(model_row[c], float) and np.isnan(model_row[c]))]

    preprocess_args = set(inspect.getargs(get_inputs_to_model.__code__).args).intersection(model_row.index.values)
    
    preprocess_dict = model_row[preprocess_args].to_dict()

    if 'win_lim' in preprocess_dict.keys():
        preprocess_dict['win_lim'] = literal_eval(preprocess_dict['win_lim'])

    model_dict = model_row[model_args].to_dict()

    model_dict = {k[6:]:v for k,v in model_dict.items()}
    if 'min_samples_leaf' in model_dict.keys():
        model_dict['min_samples_leaf'] = int(model_dict['min_samples_leaf'])

    return preprocess_dict, model_dict

def test_model(model_row, train_peak_df, test_peak_df, input_info, df):
    '''Fit best model for each reference frame on test data and record results'''

    lfads_params = model_row['lfads_params']
    lfads_filename = os.path.dirname(__file__) + '/../data/model_output/' + \
                        '_'.join([file_root, lfads_params, 'all.h5'])
    with h5py.File(lfads_filename, 'r+') as h5file:
        co = h5file['controller_outputs'][:]
        dt = utils.get_dt(h5file, input_info)
        trial_len = utils.get_trial_len(h5file, input_info)

    preprocess_dict, model_dict = get_row_params(model_row)

    model = estimator_dict[model_row['estimator']]
    if isinstance(model, MultiOutputRegressor):
        model.estimator.set_params(**model_dict)
    else:
        model.set_params(**model_dict)

    preprocess_dict.pop('min_win_start')
    preprocess_dict.pop('max_win_stop')
        
    X_train, y_train = get_inputs_to_model(train_peak_df, co, trial_len, dt, df, **preprocess_dict)
    X_test, y_test = get_inputs_to_model(test_peak_df, co, trial_len, dt, df, **preprocess_dict)
    model.fit(X_train, y_train)
    nsplits = 5
    scores = []
    y_pred = model.predict(X_test)
    for i in range(nsplits):
        y_test_sample, y_pred_sample = resample(y_test, y_pred)
        scores.append(r2_score(y_test_sample, y_pred_sample))

    score_mean = np.mean(scores)
    score_std = np.std(scores)    
    #score = model.score(X_test, y_test)

    return score_mean, score_std

if __name__=='__main__':
    output_filename = snakemake.input[0]
    params = dict(snakemake.params) #paramaters take the max performance over and fit a model
    event_type = snakemake.wildcards.event_type
    best_model_filename = snakemake.output[0]

    # output_filename = '../data/peaks/params_search_targets-not-one.csv'
    # params = {"use_rates":[False]}#dict(snakemake.params) #paramaters take the max performance over and fit a model
    # event_type = 'targets-not-one'#snakemake.wildcards.event_type
    # best_model_filename = 'controller_best_models_targets-not-one.csv'#snakemake.output[0]

    output = pd.read_csv(output_filename)
    used_estimator = cfg['used_estimator']
    output = output.query('estimator == @used_estimator')
    run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))
    datasets = [(v['name'],k) for k,v in run_info.items()]
    param_rows = []
    scores = []
    score_stds = []
    for i, (dset, file_root) in enumerate(datasets):
        data_filename = os.path.dirname(__file__) + '/../data/intermediate/' + file_root + '.p'
        inputInfo_filename = os.path.dirname(__file__) + '/../data/model_output/' + \
                                '_'.join([file_root, 'inputInfo.mat'])
        train_filename = os.path.dirname(__file__) + '/../data/peaks/' + \
                                '_'.join([file_root, '%s_train.p'%event_type])
        test_filename = os.path.dirname(__file__) + '/../data/peaks/' + \
                                '_'.join([file_root, '%s_test.p'%event_type])

        train_df = pd.read_pickle(train_filename)
        test_df = pd.read_pickle(train_filename)

        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        
        dset_out = output.query('dataset==@dset')
        #get max performance for each combination of parameters given
        for param_set in itertools.product(*params.values()):
            param_dict = {k:p for k,p in zip(params.keys(), param_set)}
            query_str = ' & '.join(['%s==\"%s\"'%(k,p) if isinstance(p, str) else '%s==%s'%(k,p) for k,p in param_dict.items()])
            max_out = dset_out.loc[dset_out.query(query_str)['total_test_score'].idxmax()]
            score_mean, score_std = test_model(max_out,
                                     train_df, test_df, input_info, df)

            param_rows.append(pd.DataFrame(max_out).T)
            scores.append(score_mean)
            score_stds.append(score_std)

    best_models = pd.concat(param_rows, ignore_index=True)
    best_models['final_held_out_score'] = scores
    best_models['final_held_out_std'] = score_stds
    best_models.to_csv(best_model_filename)