import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from extract_autonomous import *

bin_size = 10

def fit_system(intervals, n_components):
    print("fit function entered")
    y_prime = []
    y = []
    pca = PCA(n_components=n_components)
    X = np.concatenate(intervals)
    pca.fit(X)
    print("pca fit")
    for interval,t in iterate_intervals(intervals):
        transformed = pca.transform(interval)
        y_prime.append(np.gradient(transformed, t, axis=0))
        y.append(transformed)

    y = np.concatenate(y)
    y_prime = np.concatenate(y_prime)
    A = np.linalg.lstsq(y, y_prime)[0]
    print("system fit")
    return A, pca

def iterate_intervals(intervals):
    '''generator that iterates over intervals, returning a tuple of each interval and the corresponding time labels for each point'''
    bin_size_s = bin_size/1000
    for y in intervals:
        t = np.arange(0, len(y)) * bin_size_s
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
    counts = segment_spike_counts(df, co, bin_size, smooth=True)
    lim = int(min_segment/(bin_size/1000))
    # counts = np.array([np.sqrt(c[:lim,:]) for c in counts])
    # counts = np.transpose(counts, (1,0,2))
    # counts -= counts.mean(axis=0)
    print("counts segmented")
    #counts = np.transpose(counts, (1,0,2))
    train_counts, test_counts = split_segments(counts)

    A, pca = fit_system(train_counts, n_components)
    predicted = predict_system(test_counts, A, pca)

    test_counts = [pca.transform(t) for t in test_counts]
    return predicted, test_counts, A, pca

def variance_explained(predicted, actual):
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    return 1 - np.sum((predicted-actual)**2)/np.sum((actual-np.mean(actual))**2)

def one_bin_predict(intervals, A, pca):
    def f(t, y):
        return A.dot(y)
    
    actual = [pca.inverse_transform(interval[1:,:]) for interval in intervals]
    predicted = []
    for interval in intervals:
        trial_predict = np.zeros((interval.shape[0] - 1, interval.shape[1]))
        for i in range(interval.shape[0]-1):
            solution = solve_ivp(f, (0, bin_size/1000), interval[i,:])
            trial_predict[i,:] = solution.y[:,-1]
        
        predicted.append(pca.inverse_transform(trial_predict))

    return predicted, actual

# if __name__=='__main__':
#     trial_type = 'all'
#     data_filename = '../data/intermediate/rockstar.p'
#     lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_8QTVEk_%s.h5"%trial_type

#     with h5py.File(lfads_filename, 'r') as h5file:
#         co = np.array(h5file['controller_outputs'])

#     df = pd.read_pickle(data_filename)

#     predicted, actual, A, pca = fit_predict(df, co, 6)

