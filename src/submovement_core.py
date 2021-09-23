import numpy as np
from scipy import signal
import yaml
import os
from scipy.signal import savgol_filter
import sys
sys.path.insert(0,'.')
import utils
import segment_submovements as ss
from importlib import reload
reload(utils)

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))

def speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets, 
                angle_win=cfg['angle_win'], angle_thresh=cfg['angle_threshold']):
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
    angle_diffs = get_angle_diffs(minima, x_vel, y_vel, angle_win)
    angle_thresh = np.pi * angle_thresh/180 #convert to radians
    all_prominences = get_right_prominence(minima, speed)
    idx_to_remove = np.array([i for i in range(len(minima)) if 
                              all_prominences[i] < prominence and angle_diffs[i] < angle_thresh])
    # idx_to_remove = np.array([i for i in range(len(minima)) if 
    #                            all_prominences[i] < prominence])

    if np.all([i in [0, 10, 1120, 1686, 2550, 3482, 4306] for i in idx_targets]):
        import pdb;pdb.set_trace()
    minima = np.delete(minima, idx_to_remove)
    all_prominences = np.delete(all_prominences, idx_to_remove)
        
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

def speed_minima(x_vel, y_vel, threshold, prominence, distance, idx_targets, 
                angle_win=cfg['angle_win'], angle_thresh=cfg['angle_threshold'],
                post_target=cfg['exclude_movement_post_target_win']):
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
    
    # removes minima immediately after target
    # idx_post_target = [i for i in range(len(minima)) if 
    #                     np.any((0 < minima[i] - idx_target) & 
    #                             (minima[i] - idx_target < post_target))]
    # removes minima below right-prominence threshold
    angle_diffs = get_angle_diffs(minima, x_vel, y_vel, angle_win)
    angle_thresh = np.pi * angle_thresh/180 #convert to radians
    all_prominences = get_right_prominence(minima, speed)
    # idx_to_remove = np.array([i for i in range(len(minima)) if 
    #                           all_prominences[i] < prominence and angle_diffs[i] < angle_thresh])
    idx_to_remove = [i for i in range(len(minima)) if all_prominences[i] < prominence]
    
    # removes minima immediately after target
    # idx_post_target = [i for i in range(len(minima)) if 
    #                     np.any((0 < minima[i] - idx_target) & 
    #                             (minima[i] - idx_target < post_target))]
    #idx_to_remove += idx_post_target
    minima = np.delete(minima, idx_to_remove)
    all_prominences = np.delete(all_prominences, idx_to_remove)
        
    # array of minima indices, with space added in at target boundaries so 
    # submovements on other sides of the boundary aren't removed
    target_spaced_minima = np.array([m + distance*np.sum(idx_targets<m) for m in minima])

    # removes local minima until distance criteron is satisfied 
    while any(np.diff(target_spaced_minima) < distance):
        idx = np.argmax(np.diff(target_spaced_minima)< distance)
        remove_candidates = [idx, idx + 1]
        candidate_lp = get_left_prominence(minima[remove_candidates], speed)
        candidate_rp = all_prominences[remove_candidates]
        if (candidate_rp[1] < cfg['right_prominence_ratio']*candidate_rp[0] and
            candidate_lp[1] < cfg['left_prominence_thresh']):
            to_remove = remove_candidates[1]
        else:
        # removing peak with smaller prominence
            to_remove = remove_candidates[np.argmin(all_prominences[remove_candidates])]

        target_spaced_minima = np.delete(target_spaced_minima, to_remove)
        minima = np.delete(minima, to_remove)
        all_prominences = np.delete(all_prominences, to_remove)

    return minima

def get_angle_diffs(minima, x_vel, y_vel, win):
    #normalizing velocity vectors
    vel = np.vstack([x_vel, y_vel])
    x_vel, y_vel = vel / np.linalg.norm(vel, axis=0)

    angles1 = [np.arctan2(y_vel[max(m-win,0):m].mean(), x_vel[max(m-win,0):m].mean()) for m in minima]
    angles2 = [np.arctan2(y_vel[m:m+win].mean(), x_vel[m:m+win].mean()) for m in minima]

    angle_diff = [ss.angle_difference(a1, a2) for a1, a2 in zip(angles1, angles2)]

    return angle_diff

def get_right_prominence(minima, speed):
    maxima,_ = signal.find_peaks(speed)
    maxima = np.append(maxima, len(speed)-1) #in case of no local max after local min
    prominences = np.zeros(len(minima))
    for i, mi in enumerate(minima):
        next_peak = maxima[maxima > mi][0]
        prominences[i] = speed[next_peak] - speed[mi]

    return prominences

def get_left_prominence(minima, speed):
    maxima,_ = signal.find_peaks(speed)
    maxima = np.insert(maxima, 0, 0) #in case of no local max before local min
    prominences = np.zeros(len(minima))
    for i, mi in enumerate(minima):
        prev_peak = maxima[maxima < mi][-1]
        prominences[i] = speed[prev_peak] - speed[mi]

    return prominences