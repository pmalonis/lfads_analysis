from ast import literal_eval
import numpy as np
import h5py
import pandas as pd
from scipy import io
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from model_evaluation import get_row_params, estimator_dict
from optimize_target_prediction import get_inputs_to_model
import yaml
import os
from sklearn.metrics import explained_variance_score
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLMResults
from astropy.stats.circstats import circvar

def train_test_score(idx_train, idx_test, model, X, y, ja=False):
    model.fit(X[idx_train], y[idx_train])
    y_pred = model.predict(X[idx_test])
    y_true = y[idx_test]
    
    return explained_variance_score(y_true, y_pred, multioutput='variance_weighted')
    if ja:
        theta_pred = np.arctan2(y_pred[:,1], y_pred[:,0])
        theta_true = np.arctan2(y_true[:,1], y_true[:,0])

        d_theta_shoulder = np.arctan2(np.sin(y_pred[:,0]-y_true[:,0]), 
                                      np.cos(y_pred[:,0]-y_true[:,0]))
        d_theta_elbow = np.arctan2(np.sin(y_pred[:,1]-y_true[:,1]), 
                                      np.cos(y_pred[:,1]-y_true[:,1]))
            
        return np.mean([1-np.mean(d_theta_shoulder**2)/circvar(y_true[:,0]), 
                        1-np.mean(d_theta_elbow**2)/circvar(y_true[:,1])])
    else:

        theta_pred = np.arctan2(y_pred[:,1], y_pred[:,0])
        theta_true = np.arctan2(y_true[:,1], y_true[:,0])

        d_theta = np.arctan2(np.sin(theta_pred - theta_true), np.cos(theta_pred - theta_true)) 
        
        return 1-np.mean(d_theta**2)/circvar(theta_true)

def split_score(idx_a, idx_b, model, X, y, ja=False):
    score_a = train_test_score(idx_a, idx_b, model, X, y, ja)
    score_b = train_test_score(idx_b, idx_a, model, X, y, ja)
    
    return (score_a + score_b)/2

if __name__=='__main__':
    cv_repeats = 5
    remake = False
    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    cfg = yaml.safe_load(open(config_path, 'r'))

    run_info = yaml.safe_load(open(os.path.dirname(__file__) + '/../lfads_file_locations.yml', 'r'))
    #run_info.pop('mack')
#run_info.pop('raju-M1-no-bad-trials')
    output = pd.read_csv(os.path.dirname(__file__) + '/../data/peaks/old_params_search_firstmove.csv')

    for dset_idx, dataset in enumerate(run_info.keys()):
        peak_df = pd.read_pickle(os.path.dirname(__file__) 
                                 + '/../data/peaks/%s_firstmove_all.p'%dataset)

        #param = open(os.path.dirname(__file__) + '/../data/peaks/%s_selected_param_%s.txt'%(dataset, cfg['selection_metric'])).read()
        param = open(os.path.dirname(__file__) + '/../data/peaks/%s_selected_param_%s.txt'%(dataset, "sparse")).read().strip()
    
        df = pd.read_pickle('../data/intermediate/%s.p'%dataset)                  
        firstmove_df = pd.read_pickle('../data/peaks/%s_firstmove_all.p'%dataset)
        #corr_df = pd.read_pickle('../data/peaks/%s_corrections_all.p'%dataset)
        #maxima_df = pd.read_pickle('../data/peaks/%s_maxima_all.p'%dataset)
        input_info = io.loadmat('../data/model_output/%s_inputInfo.mat'%dataset)
        
        with h5py.File('../data/model_output/%s_%s_all.h5'%(dataset,param),'r') as h5file:
            co = h5file['controller_outputs'][:]
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        dset_name = run_info[dataset]['name']
        co_idxmax = output.query("dataset==@dset_name & ~use_rates")['total_test_score'].idxmax()
        rate_idxmax = output.query("dataset==@dset_name & use_rates")['total_test_score'].idxmax()
        co_row = output.loc[co_idxmax]
        rate_row = output.loc[rate_idxmax]
        co_preprocess_dict, co_model_dict = get_row_params(co_row)
        rate_preprocess_dict, rate_model_dict = get_row_params(rate_row)

        #rate_preprocess_dict['win_lim'] = co_preprocess_dict['win_lim']
        rate_preprocess_dict['rate_pcs'] = 2
        co_model = estimator_dict[co_row['estimator']]
        rate_model = estimator_dict[rate_row['estimator']]
        for model, model_dict in zip([co_model, rate_model], [co_model_dict, rate_model_dict]):
            if isinstance(model, MultiOutputRegressor):
                model.estimator.set_params(**model_dict)
            else:
                model.set_params(**model_dict)
            
        min_win_start = min(co_preprocess_dict['win_lim'][0], 
                            rate_preprocess_dict['win_lim'][0])
        max_win_stop = max(co_preprocess_dict['win_lim'][1], 
                           rate_preprocess_dict['win_lim'][1])
        co_preprocess_dict['min_win_start'] = min_win_start
        co_preprocess_dict['max_win_stop'] = max_win_stop
        rate_preprocess_dict['min_win_start'] = min_win_start
        rate_preprocess_dict['max_win_stop'] = max_win_stop

        if remake or not os.path.exists('%s_X_rate_sc.npy'%dataset) or not os.path.exists('%s_y_rate_sc.npy'%dataset):
            X_rate_sc, y_rate_sc = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='sc', **rate_preprocess_dict)
            np.save('%s_X_rate_sc.npy', X_rate_sc)
            np.save('%s_y_rate_sc.npy'%dataset, y_rate_sc)
        else:
            X_rate_sc, y_rate_sc = (np.load('%s_X_rate_sc.npy'%dataset), np.load('%s_y_rate_sc.npy'%dataset))

        if remake or not os.path.exists('%s_X_rate_ca.npy'%dataset) or not os.path.exists('%s_y_rate_ca.npy'%dataset):
            X_rate_ca, y_rate_ca = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='ca', **rate_preprocess_dict)
            np.save('%s_X_rate_ca.npy'%dataset, X_rate_ca)
            np.save('%s_y_rate_ca.npy'%dataset, y_rate_ca)
        else:
            X_rate_ca, y_rate_ca = (np.load('%s_X_rate_ca.npy'%dataset), np.load('%s_y_rate_ca.npy'%dataset))

        if remake or not os.path.exists('%s_X_sc.npy'%dataset) or not os.path.exists('%s_y_sc.npy'%dataset):
            X_sc, y_sc = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='sc', **co_preprocess_dict)
            np.save('%s_X_sc.npy'%dataset, X_sc)
            np.save('%s_y_sc.npy'%dataset, y_sc)
        else:
            X_sc, y_sc = (np.load('%s_X_sc.npy'%dataset), np.load('%s_y_sc.npy'%dataset))

        if remake or not os.path.exists('%s_X_ca.npy'%dataset) or not os.path.exists('%s_y_ca.npy'%dataset):
            X_ca, y_ca = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='ca', **co_preprocess_dict)
            np.save('%s_X_ca.npy'%dataset, X_ca)
            np.save('%s_y_ca.npy'%dataset, y_ca)
        else:
            X_ca, y_ca = (np.load('%s_X_ca.npy'%dataset), np.load('%s_y_ca.npy'%dataset))

        if dataset != 'mack':
            raw_data = io.loadmat(os.path.dirname(__file__) + '/../data/raw/%s.mat'%dataset)
            k1 = float(raw_data['monkey']['upper_kinarm'])
            k2 = float(raw_data['monkey']['lower_kinarm'])
            if remake or not os.path.exists('%s_X_rate_ja.npy'%dataset) or not os.path.exists('%s_y_rate_ja.npy'%dataset):
                rate_preprocess_dict['fit_direction'] = False
                X_rate_ja, y_rate_ja = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='ja', k1=k1, k2=k2, **rate_preprocess_dict)
                np.save('%s_X_rate_ja.npy'%dataset, X_rate_ja)
                np.save('%s_y_rate_ja.npy'%dataset, y_rate_ja)
            else:
                X_rate_ja, y_rate_ja = (np.load('%s_X_rate_ja.npy'%dataset), np.load('%s_y_rate_ja.npy'%dataset))

            if remake or not os.path.exists('%s_X_ja.npy'%dataset) or not os.path.exists('%s_y_ja.npy'%dataset):
                co_preprocess_dict['fit_direction'] = False
                X_ja, y_ja = get_inputs_to_model(peak_df, co, trial_len, dt, df=df, corr='ja', k1=k1, k2=k2, **co_preprocess_dict)
                np.save('%s_X_ja.npy'%dataset, X_ja)
                np.save('%s_y_ja.npy'%dataset, y_ja)
            else:
                X_ja, y_ja = (np.load('%s_X_ja.npy'%dataset), np.load('%s_y_ja.npy'%dataset))

        # X_rate_sc, y_rate_sc = get_inputs_to_model(peak_df.loc[:100], co, trial_len, dt, df=df, corr='sc', **rate_preprocess_dict)
        # X_rate_ca, y_rate_ca = get_inputs_to_model(peak_df.loc[:100], co, trial_len, dt, df=df, corr='ca', **rate_preprocess_dict)
        # X_sc, y_sc = get_inputs_to_model(peak_df.loc[:100], co, trial_len, dt, df=df, corr='sc', **co_preprocess_dict)
        # X_ca, y_ca = get_inputs_to_model(peak_df.loc[:100], co, trial_len, dt, df=df, corr='ca', **co_preprocess_dict)

        rpk = RepeatedKFold(n_splits=2, n_repeats=cv_repeats) #5x2 cross validation for testing
        scores = []
        coordinates = []
        predictors = []
        for idx_a, idx_b in rpk.split(X_rate_sc):
            score_sc = split_score(idx_a, idx_b, co_model, X_sc, y_sc[:,:2])
            score_ca = split_score(idx_a, idx_b, co_model, X_ca, y_ca[:,:2])
            score_rate_sc = split_score(idx_a, idx_b, rate_model, X_rate_sc, y_rate_sc[:,:2])
            score_rate_ca = split_score(idx_a, idx_b, rate_model, X_rate_ca, y_rate_ca[:,:2])

            scores += [score_sc, score_ca, score_rate_sc, score_rate_ca]
            coordinates += ['Shoulder-Aligned','Cartesian', 'Shoulder-Aligned', 'Cartesian']
            predictors += ['Controller', 'Controller', 'Rates', 'Rates']

            if dataset != 'mack':
                score_ja = split_score(idx_a, idx_b, co_model, X_ja, y_ja[:,:2], ja=True)
                score_rate_ja = split_score(idx_a, idx_b, rate_model, X_rate_ja, y_rate_ja[:,:2], ja=True)

                scores += [score_ja, score_rate_ja]
                coordinates += ['Joint Angle' ,'Joint Angle']
                predictors += ['Controller', 'Rates']

        scores_df = pd.DataFrame({'scores':scores,
                                  'coordinates':coordinates,
                                  'predictors':predictors})

        plt.subplot(1, len(run_info.keys()), 1+dset_idx)
        sns.pointplot(x='predictors', y='scores', hue='coordinates', data=scores_df)
        plt.title(run_info[dataset]['name'])
        plt.ylim([0.0, 0.85])
    
    plt.show()