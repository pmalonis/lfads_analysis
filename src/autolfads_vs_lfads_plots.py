import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

if __name__=='__main__':
    dt = 0.01
    with h5py.File('../data/model_output/rockstar_autolfads-split-trunc-02_all.h5') as g and h5py.File('../data/model_output/rockstar_split-rockstar-1000ms200ms-overlap-FDCWrX_all.h5') as f: 
        a, b = periodogram(g['output_dist_params'][:],axis=1,fs=1/dt)
        plt.semilogy(a[1:],b.mean((0,2))[1:]/np.sum(b.mean((0,2))[1:]))
        a, b = periodogram(f['output_dist_params'][:],axis=1,fs=1/dt)
        plt.semilogy(a[1:],b.mean((0,2))[1:]/np.sum(b.mean((0,2))[1:]))
        plt.legend(['autolfads','lfads'])
        plt.ylabel('Average Spectral Density')
        plt.xlabel('Frequency (Hz)')

        t = np.arange(0,f['output_dist_params'].shape[1]) * dt
        plt.savefig('../figures/rate_spectra.png')

        k = 0
        for i,j in [(0,0), (1,0),(0,50)]:
            plt.figure()
            plt.plot(t, g['output_dist_params'][i,:,j]/dt)
            plt.plot(t, f['output_dist_params'][i,:,j]/dt)
            plt.legend(['autolfads','lfads'])
            plt.xlabel('Time (s)')
            plt.ylabel('Rate (spks/s)')
            plt.savefig('../figures/example_%d'%k)
            k += 1