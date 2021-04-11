import numpy as np
import pandas as pd
import segment_submovements as ss
    

def circ_linear_np(y, theta):
    '''Nonparametric correlation coefficient between linear variable 
    y and circular variable theta (see Lototzis et al 2018)'''
    assert(len(y==len(theta)))
    n = len(y)
    r = np.argsort(theta[np.argsort(y)])
    beta = 2*np.pi * r / np.arange(1, n+1)
    def get_alpha(n):
        if n%2:
            return 2*np.sin(np.pi/n)**4/(1+np.cos(np.pi/n))**3
        else:
            (1 + 5 / np.tan(np.pi/n)**2 + 4/np.tan(np.pi/n)**4)**-1
    
    alpha = np.array([get_alpha(i) for i in range(1, n+1)])
    C = np.arange(1,n+1).dot(np.cos(beta))
    S = np.arange(1,n+1).dot(np.sin(beta))
    D = alpha*(C**2 + S**2)

    return D

if __name__=='__main__':
    trial_type = 'all'

    lfads_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_8QTVEk_%s.h5"%trial_type
    data_filename = "/home/pmalonis/lfads_analysis/data/intermediate/rockstar.p"
    inputInfo_filename = "/home/pmalonis/lfads_analysis/data/model_output/rockstar_inputInfo.mat"

    df = pd.read_pickle(data_filename)
    input_info = io.loadmat(inputInfo_filename)
    with h5py.File(lfads_filename) as h5file:
        co = h5file['controller_outputs'].value
        
    used_inds = utils.get_indices(input_info, trial_type)