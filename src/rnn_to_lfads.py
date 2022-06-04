import numpy as np
from scipy import io
import pickle

filename = 'fit_short_direction_transient_posture_serial_2targ_200_units.p'
# TODO match to rates in data
# two versions of autolfads
# handheld penalty range
#  
max_rate = 1000.0
dt = 0.01
rates = pickle.load(open(filename, 'rb'))['rates'].numpy()
#rates = rates[:,:,:]
rates *= (dt*max_rate)/np.max(rates)
counts = np.random.poisson(rates)
timeVecMs = np.arange(rates.shape[2])*0.01*1000
truth = rates
out = {'counts':counts, 'timeVecMs':timeVecMs, 'truth':truth}

io.savemat('../data/raw/fit_short_direction_transient_posture_serial_2targ_200_units.mat', out)