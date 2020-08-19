import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA, FactorAnalysis
from extract_autonomous import *
from scipy.stats import norm
from sklearn.metrics import r2_score
from scipy.linalg import subspace_angles

bin_size = 10

def fit_system(intervals, n_components, new_binsize=bin_size/1000):

    y_prime = []
    y = []
    pca = PCA(n_components=n_components)
    #pca = FactorAnalysis(n_components=n_components)
    pca.fit(np.concatenate(intervals))
    print("pca fit")
    for interval,t in iterate_intervals(intervals, new_binsize):
        transformed = pca.transform(interval)
        y_prime.append(np.gradient(transformed, t, axis=0))
        y.append(transformed)

    y = np.concatenate(y)
    y_prime = np.concatenate(y_prime)
    A = np.linalg.lstsq(y, y_prime, rcond=None)[0]
    A = A.T
    print("system fit")
    return A, pca

def fit_no_reduce(intervals):

    y_prime = []
    y = []
    
    for interval,t in iterate_intervals(intervals):
        y_prime.append(np.gradient(interval, t, axis=0))
        y.append(interval)

    y = np.concatenate(y)
    y_prime = np.concatenate(y_prime)
    A = np.linalg.lstsq(y, y_prime, rcond=None)[0]
    A = A.T
    print("system fit")
    return A, y, y_prime

def smooth_spikes(intervals, initial_binsize, new_binsize, std):
    assert((new_binsize/initial_binsize)%1 == 0)

    bin_samples = int(new_binsize/initial_binsize)
    bin_kernel = np.ones(bin_samples)
    binned_intervals = np.apply_along_axis(lambda x: np.convolve(x, bin_kernel, 'valid')[0::bin_samples], axis=1, arr=intervals)
    binned_intervals = 2*np.sqrt(binned_intervals + 3/8) #anscombe transformation
    kern = norm.pdf(np.arange(-3, 3, new_binsize/std))
    kern /= sum(kern)
    kern *= 1/new_binsize
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kern, 'valid'), axis=1, arr=binned_intervals)
    
    return smoothed, binned_intervals

def dmd(intervals, n_components, new_binsize=bin_size/1000):
    X = np.concatenate([interval[:-1] for interval in intervals])
    X_prime = np.concatenate([interval[1:] for interval in intervals])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    X_prime = pca.transform(X_prime)
    A = np.linalg.lstsq(X, X_prime, rcond=None)[0]

    return A, pca

def dmd_one_step_variance_explained(intervals, A, pca, new_binsize):

    actual = [interval[1:,:] for interval in intervals]
    # if type(pca) in (FactorAnalysis, PCA):
    #     actual = [pca.transform(interval[1:,:]) for interval in intervals]
    # elif type(pca) == np.ndarray:
    #     mean = np.concatenate(intervals).mean(0)
    #     actual = [pca.T.dot(interval[1:,:]-mean) for interval in intervals]
    # else:
    #     raise ValueError("pca input must be PCA, FactorAnalysis, or np.ndarray")

    predicted = []
    for interval in intervals:
        transformed = pca.transform(interval)
        trial_prediction = transformed.dot(A)
        if type(pca) == PCA:
            predicted.append(pca.inverse_transform(trial_prediction))
        elif type(pca) == np.ndarray:
            mean = np.concatenate(intervals).mean(0)
            predicted.append(pca.T.dot(trial_prediction) + mean)
        elif type(pca) == FactorAnalysis:
            predicted.append(pca.components_.T.dot(trial_prediction.T).T + pca.mean_)

    return predicted, actual

def smooth_spikes_unequal(intervals, initial_binsize, new_binsize, std):
    assert((new_binsize/initial_binsize)%1 == 0)

    bin_samples = int(new_binsize/initial_binsize)
    bin_kernel = np.ones(bin_samples)
    binned_intervals = np.array([np.apply_along_axis(lambda x: np.convolve(x, bin_kernel, 'valid')[0::bin_samples], 
                                 axis=0, arr=interval) for interval in intervals])
    binned_intervals = 2*np.sqrt(binned_intervals + 3/8)
    kern = norm.pdf(np.arange(-3, 3, new_binsize/std))
    kern /= sum(kern)
    kern *= 1/new_binsize
    smoothed = np.array([np.apply_along_axis(lambda x: np.convolve(x, kern, 'valid'), axis=0, arr=binned_interval) for binned_interval in binned_intervals])
    
    return smoothed, binned_intervals

def iterate_intervals(intervals, new_binsize=bin_size/1000):
    '''generator thats iterates over intervals, returning a tuple of each interval and the corresponding time labels for each point'''
    for y in intervals:
        t = np.arange(0, len(y)) * new_binsize
        yield y,t

def predict_system(intervals, A, pca):
    def f(t, y):
        return A.dot(y)

    predicted = []
    for interval,t in iterate_intervals(intervals):
        y = pca.transform(interval)
        predicted.append(solve_ivp(f, (0, t[-1]), y[0,:], t_eval=t).y.T)

    return predicted
    
def fit_predict(df, co, n_components):
    counts = segment_spike_counts(df, co, smooth=True)
    # lim = int(min_segment/(bin_size/1000))
    # counts = np.array([c[:lim,:] for c in counts])
    # counts = np.transpose(counts, (1,0,2))
    # counts -= counts.mean(axis=0)
    # counts = np.transpose(counts, (1,0,2))
    print("counts segmented")
    train_counts, test_counts = split_segments(counts)

    A, pca = fit_system(train_counts, n_components)
    predicted = predict_system(test_counts, A, pca)

    test_counts = [pca.transform(t) for t in test_counts]
    return predicted, test_counts, A, pca

def variance_explained(predicted, actual):
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    return 1 - np.sum((predicted-actual)**2)/np.sum((actual-np.mean(actual))**2)

def one_step_variance_explained(intervals, A, pca, new_binsize):
    def f(t, y):
        return A.dot(y)

    actual = [interval[1:,:] for interval in intervals]
    # if type(pca) in (FactorAnalysis, PCA):
    #     actual = [pca.transform(interval[1:,:]) for interval in intervals]
    # elif type(pca) == np.ndarray:
    #     mean = np.concatenate(intervals).mean(0)
    #     actual = [pca.T.dot(interval[1:,:]-mean) for interval in intervals]
    # else:
    #     raise ValueError("pca input must be PCA, FactorAnalysis, or np.ndarray")

    predicted = []
    for interval in intervals:
        if type(pca) in (FactorAnalysis, PCA):
            transformed = pca.transform(interval)
        elif type(pca) == np.ndarray:
            transformed = interval.dot(pca)

        trial_prediction = np.zeros((interval.shape[0]-1, transformed.shape[1]))
        for i in range(interval.shape[0]-1):
            solution = solve_ivp(f, (0, new_binsize), transformed[i,:])
            trial_prediction[i,:] = solution.y[:,-1]
        
        if type(pca) == PCA:
            predicted.append(pca.inverse_transform(trial_prediction))
        elif type(pca) == np.ndarray:
            mean = np.concatenate(intervals).mean(0)
            predicted.append(trial_prediction.dot(pca.T) + mean)
        elif type(pca) == FactorAnalysis:
            predicted.append(pca.components_.T.dot(trial_prediction.T).T + pca.mean_)

    return predicted, actual

def simulate_system(N, n_neurons, T=min_segment, rate_dist_scale=None):
    A = np.array([[-0.625, -20.5],[12.5,-0.625]]) #gives rotations, min segment
    def f(t, y):
        return A.dot(y)

    W = np.random.randn(int(n_neurons/1),2) #weight matrix projecting factors onto rates
    W /= np.linalg.norm(W, axis=0)
    W[:,0] = W[:,0] - W[:,0].dot(W[:,1])*W[:,1]
    W[:,0] /= np.linalg.norm(W[:,0])
    #W = np.tile(W, (20,1))

    ms_bin = 0.001
    radius = 10 #radius of initial conditions

    rates = np.zeros((N, int(T/ms_bin), n_neurons))
    samples = np.zeros_like(rates)
    factors = np.zeros((N, int(T/ms_bin), A.shape[0]))

    for i in range(N):
        t = np.arange(0, T, ms_bin)

        #getting random initial condition
        theta = np.random.rand()*2*np.pi
        init = np.array([np.cos(theta), np.sin(theta)]) * radius

        s = solve_ivp(f, (0, t[-1]), init, t_eval=t)
        factors[i,:,:] = s.y.T
        r = W.dot(s.y).T
        rates[i,:,:] = r
    
    rates -= np.min(rates)
    rates *= 20/np.max(rates)
    #rates *= 20/np.min(rates)
    if rate_dist_scale:
        rates *= np.random.exponential(scale=rate_dist_scale, size=n_neurons)

    p = rates * ms_bin
    samples = np.random.rand(*rates.shape) < p
    
    return samples, rates, factors, W

def sim_corr_components(new_binsize, std):
    
    s,r,f,W = simulate_system(200,100,T=1)
    smoothed=smooth_spikes(s, .001, new_binsize, std)
    A,pca=fit_system(smoothed,2)
    
    return np.corrcoef(W[:,0], pca.components_[0]), A, pca, smoothed, W

def sim_predict(new_binsize, std):
    s,r,f,W = simulate_system(100,100,T=1)
    smoothed=smooth_spikes(s, .001, new_binsize, std)
    A,pca=fit_system(smoothed,2)    

    p,a=one_step_variance_explained(smoothed,A,pca)

    return r2_score(np.concatenate(a),np.concatenate(p))

# if __name__=='__main__':
#     trial_type = 'all'
#     data_filename = '../data/intermediate/rockstar.p'
#     lfads_filename = "/home/pmalonis/226_figs/rockstar_8QTVEk_%s.h5"%trial_type

#     with h5py.File(lfads_filename, 'r') as h5file:
#         co = np.array(h5file['controller_outputs'])

#     df = pd.read_pickle(data_filename)
#     predicted, actual, A, pca = fit_predict(df, co,6)