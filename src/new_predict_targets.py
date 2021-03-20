import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_inputs_to_model(peak_df, co, dt, reference_frame, rused_inds=None):
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

def get_inputs_to_model_control(df, peak_df, n_components, reference_frame):
    