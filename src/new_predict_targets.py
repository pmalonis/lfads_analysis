import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBoostRegressor, XGBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '.')
import utils
import timing_analysis as ta

np.random.seed(50818)
config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
cv = cfg['target_prediction_cv_splits']
# def predict_target_pos(regressor, predictor, target_x, target_y):
#     '''
#     '''
#     return cross_val_score(regressor, predictor, target_pos, cv=5)

def predict_target(regressor, predictor, target_x, target_y, pos=True):
    '''

    '''
    if pos:
        target_pos = np.vstack((target_x, target_y))
    else:
        y = np.arctan2(target_y,target_x)

    return cross_val_score(regressor, predictor, y, cv=cv)

def plot_with_control(regressor, predictor, control, x_target, y_target, x_hand, y_hand, pos=False):

    score_shoulder = predict_target(regressor, predictor, x_target, y_target, pos)
    control_shoulder = predict_target(regressor, control, x_target, y_target, pos)
    score_hand = predict_target(regressor, predictor, x_target-x_hand, y_target-y_hand, pos)
    control_hand = predict_target(regressor, control, x_target-x_hand, y_target-y_hand, pos)

    df=pd.DataFrame(data={'r^2':np.concatenate([score_shoulder, control_shoulder, score_hand, control_hand]),
                          'coordinate':['shoulder']*10+['hand']*10,'predictor':['controller']*5+['spikes']*5+['controller']*5+['spikes']*5})
    ax = sns.pointplot(x='coordinate',y='r^2',hue='predictor',data=df)

    return ax

def get_inputs_to_model(peak_df, co, dt, reference_frame, used_inds=None):
    #removing targets for which we don't have a full window of controller inputs
    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - cfg['post_target_win_stop'])]
    
    if used_inds is None:
        assert(peak_df.index[-1][0] + 1 == co.shape[0])
        used_inds = range(co.shape[0])
    
    k = 0 # target counter
    win_size = int((cfg['post_target_win_stop'] - cfg['post_target_win_start'])/dt)
    X = np.zeros((peak_df.shape[0], win_size*co.shape[2]))
    for i in used_inds:
        trial_peak_df = peak_df.loc[i]
        target_times = trial_peak_df.index
        for target_time in target_times:
            idx_start = int((target_time + cfg['post_target_win_start'])/dt)
            idx_stop = int((target_time + cfg['post_target_win_stop'])/dt)
            X[k,:] = co[i,idx_start:idx_stop,:].flatten()
            k += 1

    if reference_frame == 'shoulder':
        y = peak_df[['target_x', 'target_y']]
    elif reference_frame == 'hand':
        y = peak_df[['target_x', 'target_y']] - peak_df[['x', 'y']]

    return X, y

def get_inputs_to_model_control(df, peak_df, n_components, reference_frame, trial_len, spike_dt=0.001, lfads_dt=0.01, used_inds=None):
    peak_df = peak_df.iloc[np.where(peak_df.index.get_level_values('time') < trial_len - cfg['post_target_win_stop'])]

    if used_inds is None:
        assert(peak_df.index[-1][0] + 1 == co.shape[0])
        used_inds = range(co.shape[0])

    win = lfads_dt/spike_dt
    midpoint_idx = int((win-1)/2)
    nneurons = sum('neural' in c for c in df.columns)
    all_smoothed = np.zeros(len(used_inds), int(trial_len/dt), nneurons)
    for i in used_inds:
        smoothed = data.loc[i].neural.rolling(window=300, min_periods=1, win_type='gaussian', center=True).mean(std=50)
        smoothed = smoothed.loc[:trial_len].iloc[midpoint_idx::win]
        smoothed = smoothed.loc[np.all(smoothed.notnull().values, axis=1),:].values #removing rows with all values null (edge values of smoothing)
        all_smoothed[i,:,:] = smoothed

    pca = PCA(n_components=n_components)
    pca.fit(np.vstack(all_smoothed))

    k = 0 # target counter
    win_size = int((cfg['post_target_win_stop'] - cfg['post_target_win_start'])/dt)
    X = np.zeros((peak_df.shape[0], win_size*co.shape[2]))
    for i in used_inds:
        trial_peak_df = peak_df.loc[i]
        target_times = trial_peak_df.index
        for target_time in target_times:
            idx_start = int((target_time + cfg['post_target_win_start'])/dt)
            idx_stop = int((target_time + cfg['post_target_win_stop'])/dt)
            X[k,:] = pca.transform(all_smoothed[i,idx_start:idx_stop,:])
            k += 1

    if reference_frame == 'shoulder':
        y = peak_df[['target_x', 'target_y']]
    elif reference_frame == 'hand':
        y = peak_df[['target_x', 'target_y']] - peak_df[['x', 'y']]

    return X, y


def get_reverse_model_inputs(peak_df, used_inds=None, pos=False):
    '''
    peak_df can be created with controller outputs or spikes
    '''

    pkdf = peak_df.copy()
    pkdf['input'] = ''

    pkdf.loc[pkdf.latency_0.notnull() & pkdf.latency_1.isnull(), 'input'] = 'input_1'
    pkdf.loc[pkdf.latency_0.isnull() & pkdf.latency_1.notnull(), 'input'] = 'input_2'
    pkdf.loc[pkdf.latency_0.notnull() & pkdf.latency_1.notnull(), 'input'] = 'both'
    pkdf.loc[pkdf.latency_0.isnull() & pkdf.latency_1.isnull(), 'input'] = 'neither'

    if pos==False:
        X = np.concatenate([pkdf[['target_x','target_y']], pkdf[['target_x', 'target_y']].values - pkdf[['x','y']].values], axis=1)
    else:
        #computing angles
        X = np.concatenate([np.arctan2(pkdf[['target_x','target_y']].values.T), 
                            np.arctan2(*(pkdf[['target_x', 'target_y']].values - pkdf[['x','y']].values).T)], axis=1)
    y = pkdf['input'].values

    return X, y

if __name__=='__main__':
    datasets = ['rockstar', 'raju', 'mk08011M1m_mack_RTP']
    params = ['8QTVEk', '2OLS24', '2OLS24']
    win_start = 0
    win_stop = 0.5
    min_height_list = [['0.3', '0.3'], ['0.3', '0.3'], ['0.3', '0.3']]
    for dataset, param, min_heights in zip(datasets, params, min_height_list):
        data_filename = '../data/intermediate/' + dataset + '.p'
        lfads_filename = '_'.join([dataset, param, 'all.h5'])
        inputInfo_filename = '_'.join([dataset, 'inputInfo.mat'])
        
        df = data_filename = pd.read_pickle(data_filename)
        input_info = io.loadmat(inputInfo_filename)
        with h5py(lfads_filename) as h5file:
            co = h5file['controller_outputs']
            dt = utils.get_dt(h5file, input_info)
            trial_len = utils.get_trial_len(h5file, input_info)

        used_inds = range(df.index[-1][0] + 1)
        peak_df = ta.get_peak_df(df, co, min_heights, dt, used_inds, trial_len, win_start=win_start, win_stop=win_stop)
        X, _ = get_inputs_to_model(peak_df, co, dt, 'shoulder', used_inds=None)
        hand_target = peak_df[['target_x', 'target_y']] - peak_df[['x', 'y']]
        shoulder_target = peak_df[['target_x', 'target_y']]
        model = RandomForestRegressor()
        X_control,_ = get_inputs_to_model_control(df, peak_df, n_components, 'shoulder', trial_len, spike_dt=0.001, lfads_dt=0.01)
        plot_with_control(model, X, X_control, peak_df['target_x'], peak_df['target_y'],  
                          peak_df['x'], peak_df['y'], pos=True)
        