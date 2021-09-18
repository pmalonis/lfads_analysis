import numpy as np
from scipy import signal
import yaml
import os
from scipy.signal import savgol_filter
import sys
sys.path.insert(0,'.')
import utils
from importlib import reload
reload(utils)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets):
    '''Calculates submovements from velocites, based on 
    speed profile
    
    Parameters
    x_vel: x-coordinate of velocity
    y_vel: y-coordinate of velocity
    threshold: threshold above which speed minima will not be returned
    prominence: minimum size of the following movement from trough to peak
    distance: minimum distance between submovements. this is ignored when 
                one submovement occurs before a target is hit and the other
                occurs after
    idx_target: indices of x_vel and y_vel representing times when targets
                appear

    Returns:
    minima: local speed minima that fulfill criteria
    '''

    speed = utils.get_speed(x_vel, y_vel)
    minima, _  = signal.find_peaks(-speed,
                                    height=-threshold)
    
    # removes minima below right-prominence threshold
    all_prominences = get_right_prominence(minima, speed)
    idx_too_small = np.array([i for i in range(len(minima)) if 
                               all_prominences[i] < prominence])
    minima = np.delete(minima, idx_too_small)
    all_prominences = np.delete(all_prominences, idx_too_small)
        
    # array of minima indices, with space added in at target boundaries so 
    # submovements on other sides of the boundary aren't removed
    target_spaced_minima = np.array([m + distance*np.sum(idx_targets<m) for m in minima])
    
    # removes local minima until distance criteron is satisfied 
    while any(np.diff(target_spaced_minima) < distance):
        idx = np.argmax(np.diff(target_spaced_minima)< distance)
        remove_candidates = [idx, idx + 1]
        # removing peak with smaller prominence
        to_remove = remove_candidates[np.argmin(all_prominences[remove_candidates])]
        target_spaced_minima = np.delete(target_spaced_minima, to_remove)
        minima = np.delete(minima, to_remove)
        all_prominences = np.delete(all_prominences, to_remove)

    return minima

def get_right_prominence(minima, speed):
    maxima,_ = signal.find_peaks(speed)
    maxima = np.append(maxima, len(speed)-1) #in case of no local max after local min
    prominences = np.zeros(len(minima))
    for i, mi in enumerate(minima):
        next_peak = maxima[maxima > mi][0]
        prominences[i] = speed[next_peak] - speed[mi]

    return prominences