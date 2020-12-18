import pylds
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from numpy.linalg import inv

def evaluate_lds(counts, model):
    '''evaluate one step variance explained for pylds model'''
    actual = [count[1:,:] for count in counts]
    predicted = []
    states = []
    for trial in counts:
        p = np.zeros((trial.shape[0]-1, trial.shape[1]))
        for i in range(trial.shape[0]-1):
            smoothed = model.smooth(trial[:i+1])
            x,_,_,_ = np.linalg.lstsq(model.C, smoothed[-1,:], rcond=None)
            p[i,:] = model.C.dot(model.A.dot(x))

        predicted.append(p)
        states.append(x)

    r2 = r2_score(np.concatenate(predicted), np.concatenate(actual))

    return r2, predicted, actual, states

def evaluate_lds_difference(counts, model):
    '''evaluate one step variance explained for pylds model'''
    
    predicted = []
    states = []
    for trial in counts:
        p = np.zeros((trial.shape[0]-1, trial.shape[1]))
        for i in range(trial.shape[0]-1):
            smoothed = model.smooth(trial[:i+1])
            x,_,_,_ = np.linalg.lstsq(model.C, smoothed[-1,:], rcond=None)
            p[i,:] = model.C.dot(model.A.dot(x)) - smoothed[-1,:]

        predicted.append(p)
        states.append(x)

    actual = [count[1:,:] - count[:-1,:] for count in counts]
    r2 = r2_score(np.concatenate(predicted), np.concatenate(actual))

    return r2, predicted, actual, states

def long_term_predict(counts, model, n_start, n_end):
    '''predict rest of trial from start of trial. number of steps to predict in the start
    is given by n_start. number of steps to count toward prediction given by n_end'''

    
    predicted = []
    states = []
    for trial in counts:
        p = np.zeros((trial.shape[0]-1, trial.shape[1]))
        
        smoothed = model.smooth(trial[:i+1])
        x,_,_,_ = np.linalg.lstsq(model.C, smoothed[-1,:], rcond=None)
        p[i,:] = model.C.dot(model.A.dot(x))

        predicted.append(p)
        states.append(x)

    actual = [count[1:,:] for count in counts]
    r2 = r2_score(np.concatenate(predicted), np.concatenate(actual))

    return r2, predicted, actual, states

# def evaluate_rates(counts, model, rates):
#     for trial in counts:
#         model.smooth(trial)

def evaluate_glds(counts, model):
    '''evaluate one step variance explained for pylds model'''
    actual = [count[1:,:] for count in counts]
    predicted = []
    p_0 = np.zeros(model.d_latent)
    V_0 = np.eye(model.d_latent) * 1e10
    for trial in counts:
        p = np.zeros((trial.shape[0]-1, trial.shape[1]))
        for i in range(trial.shape[0]-1):
            _,smoothed,_,_,_ = model.filter(trial[:i+1], p_0, V_0)
            p[i,:] = model.C.dot(model.A.dot(smoothed[-1,:]))

        predicted.append(p)

    r2 = r2_score(np.concatenate(predicted), np.concatenate(actual))

    return r2, predicted,actual

def filter_explained_variance(counts, model):
    return

def simulated_gen_counts(bin_size):
    pass

def center_out_gen_counts(dataset, min_t = .8):#, bin_size):
    '''preprocesses center out model
    
    dataset: xarray dataset
    bin_size: bin size for counts, in seconds'''
    
    #assert(bin_size%dataset.dt == 0) #making sure bin_size is multiple of raw data bin size
    #bin_factor = int(bin_size/dataset.dt) #number of data bins to combine
    #binned = dataset.neural.rolling(dict(time=bin_factor)).sum().astype(np.int8)
    #binned = binned[dict(time=slice(None,None,bin_factor))]
    binned=dataset.neural
    counts=[binned.loc[i, dataset.stmv[i]:dataset.endmv[i], :].values 
    for i in dataset.trial if (dataset.endmv[i]-dataset.stmv[i])>min_t]
    
    return counts

def regression_to_full_model(A, pca):
    pass

def kalman_ratio(obs, params):

    '''returns ratio of dynamic to innovative process contributions to Kalman 
    filter'''

    A, C, Q, R = params

    d_latent = A.shape[0]
    d_obs = obs.shape[1] 

    y = obs
    T = y.shape[0] #length of time series
    x = np.zeros((T, d_latent)) # will contain E(x_t|y_1,...,y_t)
    V = np.zeros((T, x.shape[1], x.shape[1])) # will contain Var(x_t|y_1,...,y_t)
    bel_x = np.zeros(d_latent) # initial mean belief
    static_est_states = np.array([np.linalg.lstsq(C, y[i], rcond=None)[0] for i in range(T)])
    initial_var_est = np.var(static_est_states, axis=0)
    bel_V = np.diag(initial_var_est)
    dynamic = np.zeros(T-1)
    innovative = np.zeros(T-1)
    for i in range(T):
        #import pdb;pdb.set_trace()
        K = bel_V.dot(C.T).dot(inv(C.dot(bel_V).dot(C.T) + R))
        #import pdb; pdb.set_trace()
        x[i,:] = bel_x +  K.dot(y[i] - C.dot(bel_x))
        V[i,:,:] = bel_V - K.dot(C).dot(bel_V)
        
        if i > 0:
            dynamic[i-1] = np.linalg.norm(bel_x - x[i-1,:])
            innovative[i-1] = np.linalg.norm(K.dot(y[i] - C.dot(bel_x)))
        #update beliefs for next sample
        
        if i < T - 1:
            bel_x = A.dot(x[i])
            bel_V = A.dot(V[i,:,:]).dot(A.T) + Q

    y_filtered = x.dot(C.T)
    ratio = np.mean(dynamic/(innovative + dynamic))

    return ratio

