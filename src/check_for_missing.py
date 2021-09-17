import pandas as pd
import numpy as np

for dataset in ['rockstar','mack','raju-M1-no-bad-trials']:
    firstmove_filename = '../data/peaks/%s_firstmove_all.p'%dataset
    corrections_filename = '../data/peaks/%s_corrections_all.p'%dataset
    new_corrections_filename = '../data/peaks/%s_new-corrections_all.p'%dataset
    corrections_filename = '../data/peaks/%s_2new-firstmove_all.p'%dataset
    new_corrections_filename = '../data/peaks/%s_new-firstmove_all.p'%dataset
    fm = pd.read_pickle(firstmove_filename)
    c = pd.read_pickle(corrections_filename)
    nc = pd.read_pickle(new_corrections_filename)
    
    fm_a = np.array(list(fm.index.values))
    c_a = np.array(list(c.index.values))
    nc_a = np.array(list(nc.index.values))

    dups = np.array([np.any(np.all(c_a[i] ==fm_a,axis=1)) for i in range(c_a.shape[0])])
    in_nc = np.array([np.any(np.all(c_a[i] ==nc_a,axis=1)) for i in range(c_a.shape[0])])
    # print(sum(~dups & ~in_nc)/len(dups))
    # print(nc.shape[0]+sum(dups))
    # print(c.shape[0])
    print((c.shape[0]-nc.shape[0])/c.shape[0])