import pandas as pd
import numpy as np

for dataset in ['rockstar','mack','raju-M1-no-bad-trials']:
    firstmove_filename = '../data/peaks/%s_new-firstmove_all.p'%dataset
    #corrections_filename = '../data/peaks/%s_new-corrections_all.p'%dataset
    corrections_filename = '../../backup.p'
    fm = pd.read_pickle(firstmove_filename)
    c = pd.read_pickle(corrections_filename)
    
    fm_a = np.array(list(fm.index.values))
    c_a = np.array(list(c.index.values))
    dups = [np.any(np.all(c_a[i] == fm_a,axis=1)) 
            for i in range(c_a.shape[0])]
    print(sum(dups)/len(dups))
    print(fm.shape[0])