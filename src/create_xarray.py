import numpy as np
import xarray as xr
from scipy.io import loadmat
from scipy.interpolate import interp1d
from convert_to_pandas import filter_kinematics

def get_neurons(mat_data):
    '''returns list of area for each neuron. neurons are '''
    neurons = [k for k in mat_data.keys() if 'Chan' in k]
    neuron_area = []

    areas = [k.strip('chans') for k in mat_data.keys() if 'chans' in k and k != 'chans']
    for neuron in neurons:
        this_neuron_area = ''
        for area in areas:
            if np.any(['Chan%03d'%c in neuron for c in mat_data['%schans'%area].flatten()]):
                this_neuron_area = area
                break

        neuron_area.append(this_neuron_area)

    return neurons, neuron_area

if __name__=='__main__':
    filename = snakemake.input[0]
    bin_size = 0.002
    data = loadmat(filename)
    min_spikes = 100 #drop neurons with fewer spikes

    cond_keys = [k for k in data.keys() if 'deg' in k and k.count('_')==1]
    
    event_names = ['instr', 'go', 'stmv', 'endmv'] #events, in order
    ref = 'stmv' #event to serve as time 0 of time coordinate

    events = {ev:np.array([]) for ev in event_names} #list of times of reference events
    angles = np.array([])
    for k in cond_keys:
        cond_events = {en:data[k + '_' + en].flatten() for en in event_names}
        assert(len(set([len(cond_events[en]) for en in event_names])) == 1) #makes sure there is one event for each trial in condition
        for en in event_names: 
            events[en] = np.append(events[en], cond_events[en])

        cond_n_trials = len(cond_events[ref]) 
        angle = int(k.split('_')[1].strip('deg'))
        repeated_angles = [angle] * cond_n_trials
        angles = np.append(angles, repeated_angles)

    #sort events 
    sorted_idx = np.argsort(events[event_names[0]])
    for en in event_names:
        events[en] = events[en][sorted_idx]

    angles = angles[sorted_idx]

    max_start = np.max(events[event_names[0]]-events[ref])
    max_stop = np.max(events[event_names[-1]]-events[ref])

    neuron_names, neuron_area = get_neurons(data)

    n_trials = len(angles)
    n_bins = np.ceil((max_stop - max_start)/bin_size).astype(int)
    n_neurons = len(neuron_names)

    time_labels = np.arange(max_start, max_stop, bin_size) + bin_size/2

    neural = xr.DataArray(np.zeros((n_trials, n_bins, n_neurons)), 
                          dims=['trial', 'time', 'neuron'], 
                          coords= {'trial':range(n_trials), 
                                   'time':time_labels, 
                                   'neuron':range(n_neurons)})
    kinematics = xr.DataArray(np.zeros((n_trials, n_bins, 2)), 
                              dims=['trial', 'time', 'variable'], 
                              coords= {'trial':range(n_trials), 
                                       'time':time_labels, 
                                       'variable':range(2)})

    interp_x = interp1d(data['x'][:,0], data['x'][:,1])
    interp_y = interp1d(data['y'][:,0], data['y'][:,1])
    for i in range(n_trials):
        #neural
        trial_start = events[event_names[0]][i]
        trial_stop = events[event_names[-1]][i]
        bins = np.arange(max_start + events[ref][i], max_stop + events[ref][i] + bin_size, bin_size)
        abs_times = time_labels + events[ref][i] #midpoint time of each bin relative to start of experiment
        mask = (abs_times < trial_start) | (abs_times <= trial_stop)
        for j,n in enumerate(neuron_names):
            spks = data[n][(data[n] >= trial_start) & (data[n] < trial_stop)]
            binned = np.histogram(spks, bins=bins)[0].astype(np.float)
            binned[~mask] = np.nan #-127
            neural[i,:,j] = binned

        x_raw = interp_x(abs_times)
        y_raw = interp_y(abs_times)
        x_smooth, y_smooth = filter_kinematics(x_raw, y_raw)

        x_smooth[mask] = np.nan
        y_smooth[mask] = np.nan

        kinematics[i,:,0] = x_smooth
        kinematics[i,:,1] = y_smooth

    dataset = xr.Dataset({'neural':neural, 'kinematics':kinematics})
    for k in events.keys():
        dataset[k] = ('trial', events[k]-events[ref])

    dataset.attrs['dt'] = 0.002
    spk_sum = dataset.neural.sum(['trial', 'time'])
    dataset = dataset.drop_sel(neuron = np.where(spk_sum < min_spikes)[0])
    dataset.to_netcdf(snakemake.output[0])