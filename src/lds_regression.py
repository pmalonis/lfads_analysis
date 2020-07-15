import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from extract_autonomous import *

def fit_system(intervals, bin_size, n_components):

    y_prime = []
    y = []
    pca = PCA(n_components=n_components)
    pca.fit(np.concatenate(intervals))
    print("pca fit")
    for interval,t in iterate_intervals(intervals, bin_size):
        transformed = pca.transform(np.sqrt(interval))
        y_prime.append(np.gradient(transformed, t, axis=0))
        y.append(transformed)

    y = np.concatenate(y)
    y_prime = np.concatenate(y_prime)
    A = np.linalg.lstsq(y, y_prime)[0]
    print("system fit")
    return A, pca

def iterate_intervals(intervals, bin_size):
    '''generator that iterates over intervals, returning a tuple of each interval and the corresponding time labels for each point'''
    bin_size_s = bin_size/1000
    for y in intervals:
        t = np.arange(0, len(y)) * bin_size_s
        yield y,t

def predict_system(intervals, bin_size, A, pca):
    def f(t, y):
        return A.dot(y)

    predicted = []
    for interval,t in iterate_intervals(intervals, bin_size):
        y = pca.transform(interval)
        predicted.append(solve_ivp(f, (0, t[-1]), y[0,:], t_eval=t).y.T)

    return predicted
    
def fit_predict(df, co, n_components):
    bin_size = 1
    counts = segment_spike_counts(df, co, bin_size, smooth=True)
    print("counts segmented")
    train_counts, test_counts = split_segments(counts)

    A, pca = fit_system(train_counts, bin_size, n_components)
    predicted = predict_system(test_counts, bin_size, A, pca)

    test_counts = [pca.transform(t) for t in test_counts]
    return predicted, test_counts, A

def variance_explained(predicted, actual):
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    return 1 - np.sum((predicted-actual)**2,axis=0)/np.sum((actual-np.mean(actual))**2, axis=0)


if __name__=='__main__':
    trial_type = 'all'
    data_filename = '../data/intermediate/rockstar.p'
    lfads_filename = "/home/pmalonis/226_figs/rockstar_8QTVEk_%s.h5"%trial_type

    with h5py.File(lfads_filename, 'r') as h5file:
        co = np.array(h5file['controller_outputs'])

    df = pd.read_pickle(data_filename)

    predicted, actual, A = fit_predict(df, co,6)