import pandas as pd
import numpy as np

for dataset in ['rockstar','mack','raju-M1-no-bad-trials']:
    firstmove_filename = '../data/peaks/%s_new-firstmove_all.p'%dataset
    corrections_filename = '../data/peaks/%s_new-corrections_all.p'%dataset
    fm = pd.read_pickle(firstmove_filename)
    c = pd.read_pickle(corrections_filename)
    
    fm_a = np.array(list(fm.index.values))
    c_a = np.array(list(c.index.values))
    dups = np.zeros(c_a.shape[0], dtype=bool)
    for i in range(c_a.shape[0]):
        diffs = c_a[i,1] - fm_a[:,1] < 0.3
        if np.any(((0 < diffs) & (diffs < 0.3)) & (c_a[i,0] == fm_a[:,0])):
            dups[i] == True
    
    print(sum(dups)/len(dups))