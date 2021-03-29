import subsample_analysis as sa
from glob import glob
import parse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
reload(sa)

if __name__=='__main__':
    filenames = np.sort(glob('../data/model_output/rockstar_*_trial_*_all.h5'))
    mean_1 = []
    mean_2 = []
    ntrials = []
    for filename in filenames:
        parse_result = parse.parse('{}rockstar_{ntrials:d}_trial_{}_all.h5', filename)
        ntrials.append(parse_result['ntrials'])
        with h5py.File(filename) as h5file:
            co = h5file['controller_outputs'].value

        mean_1.append(np.mean([sa.gini(co[i, :, 0]) for i in range(co.shape[0])]))
        mean_2.append(np.mean([sa.gini(co[i, :, 1]) for i in range(co.shape[0])]))

    plt.plot(ntrials, mean_1)
    plt.plot(ntrials, mean_2)
    plt.xlabel('Number of trials')
    plt.ylabel('Gini coefficient')
    plt.legend(['Input 1', 'Input 2'])