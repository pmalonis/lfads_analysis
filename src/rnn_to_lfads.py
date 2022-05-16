import numpy as np
from scipy import io
import pickle

filename = 'fit_direction_Transient_posture_serial_2targ.p'
max_rate = 100.0
dt = 0.01
rates = pickle.load(open(filename, 'rb'))['rates'].numpy()
rates = rates[:,:,18:]
rates *= (dt*max_rate)/np.max(rates)
counts = np.random.poisson(rates)
timeVecMs = np.arange(rates.shape[2])*0.01*1000
truth = rates
out = {'counts':counts, 'timeVecMs':timeVecMs, 'truth':truth}

io.savemat('../data/raw/fit_direction_Transient_posture_serial_1targ.mat', out)