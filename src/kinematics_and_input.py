import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib import animation
from scipy import io
import h5py
import os
from glob import glob

# lfads_file = 'data/model_output/rockstar_valid.h5'
# filename = 'data/raw/rockstar.mat'
# input_info_file = 'data/model_output/rockstar_inputInfo.mat'
# out_directory = '/home/pmalonis/'

lfads_file = snakemake.input[0]
filename = snakemake.input[1]
input_info_file = snakemake.input[2]
out_directory = os.path.dirname(snakemake.output[0])

os.makedirs(out_directory, exist_ok=True)

fps = 250 #frame rate at which to read data
playback_ratio = .20  # speed of playback (1 indicates real speed)
kinematic_fs = 500 #frame rame of kinematic data

input_info = io.loadmat(input_info_file)
if snakemake.wildcards.trial_type == 'train':
    used_inds = input_info['trainInds'][0] - 1
elif snakemake.wildcards.trial_type == 'valid':
    used_inds = input_info['validInds'][0] - 1
    
# used_inds = input_info['validInds'][0] - 1

def update_target(data, t):
    '''finds the next target in data dictionary and returns its index and position'''
    hit_target_idx = np.where(data['hit_target'] - t > 0)[0][0]
    t_next_target = data['hit_target'][hit_target_idx]
    idx = np.argmin(np.abs(t_next_target - data['x'][:,0]))
    target_pos = (data['x'][idx,1], data['y'][idx,1])
    return hit_target_idx, target_pos


with h5py.File(lfads_file) as h5file:
    dt = 0.01    
    trial_t = np.arange(h5file['controller_outputs'].shape[1]) * dt
    trial_len = trial_t[-1] + dt

    data = io.loadmat(filename)
    used_trials =  np.where(np.diff(data['cpl_st_trial_rew'].real, axis=1) > trial_len)[0]

    for video_idx, plotted_trial in enumerate(used_trials[used_inds]):
        t_start = data['cpl_st_trial_rew'][plotted_trial,0].real
        t_end = t_start + trial_len
        fig, ax = plt.subplots(2, figsize=(8,12))

        ax[0].set_xlim(-100, 100)
        ax[0].set_ylim(100, 300)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Trial %03d"%plotted_trial)
        cursor_ln, = ax[0].plot([], [], 'r.', markersize=8)
        target_ln, = ax[0].plot([], [], marker='s', color='b', markersize=8)

        input_lns = []
        for input_idx in range(h5file['controller_outputs'].shape[2]):
            ax[1].plot(trial_t, h5file['controller_outputs'][video_idx,:,input_idx])
            
        time_ymin = -0.75
        time_ymax = 0.75
        time_ln = ax[1].vlines(0, time_ymin, time_ymax)

        ax[1].set_ylim(time_ymin, time_ymax)
        ax[1].set_xlabel('Time(s)')

        hit_target_idx, target_pos = update_target(data, t_start)
        st_trial_idx = np.where(data['st_trial'] - t_start > 0)[0][0]
        endmv_idx = np.where(data['endmv'] - t_start > 0)[0][0]
        
        def init():
            cursor_ln.set_data([], [])
            target_ln.set_data([], [])
            return cursor_ln, target_ln

        def animate(t):
            '''Animates kinematic frame at time t by plotting cursor position and, if t
            is a time during a trial, the target position'''

            global target_pos, hit_target_idx, endmv_idx, st_trial_idx

            # plotting cursor
            if np.min(data['x'][:,0]) < t < np.max(data['x'][:,0]):
                i = np.argmin(np.abs(t - data['x'][:,0]))
                cursor_ln.set_data(data['x'][i,1], data['y'][i,1])
            else:
                raise ValueError("t out of range of experiment time")

            # # plotting target
            # if np.sum(target_pos):
            #     target_visible = True
            # else:
            #     target_visible = False

            # if target_visible:
            if t > data['hit_target'][hit_target_idx]: #if target has been hit
                # sets conditions for end of trial, which depends on whether
                # the current trial is the last one
                if st_trial_idx + 1 >= len(data['st_trial']):
                    trial_ended = t > data['endmv'][endmv_idx]
                else:
                    trial_ended = (t > data['endmv'][endmv_idx]) #The time between an endmv event and the next st_trial can be smaller than the frame rate

                if trial_ended:
                    # if trial is over, update the indices that allow finding the
                    # next trial
                    st_trial_idx = np.where(data['st_trial'] - t > 0)[0][0]
                    endmv_idx = np.where(data['endmv'] - t > 0)[0][0]
                    if t < data['st_trial'][st_trial_idx]:
                        target_pos = ([], [])
                    else:
                        hit_target_idx, target_pos = update_target(data, t)
                else:
                    # if trial is not over, update the target position and the next
                    # target acquisition time
                    hit_target_idx, target_pos = update_target(data, t)
            # else:
            #     if t > data['st_trial'][st_trial_idx]:
            #         hit_target_idx, target_pos = update_target(data, t)

            target_ln.set_data(*target_pos)

            #plotting
            lfads_idx = np.digitize(t-t_start, trial_t) - 1 #assumes t_start is trial start
            time_ln.set_segments([np.array([[trial_t[lfads_idx], time_ymin], [trial_t[lfads_idx], time_ymax]])])
            for input_idx, input_ln in enumerate(input_lns):
                input_ln.set_data(trial_t[lfads_idx], h5file['controller_outputs'][video_idx, lfads_idx, input_idx])

            return (cursor_ln, target_ln, *input_lns)

        anim = animation.FuncAnimation(fig, animate, frames=np.arange(t_start, t_end, 1./fps), init_func=init, blit=True)
        # anim = animation.FuncAnimation(fig, animate, frames=range(0, data['x'].shape[0], 10), init_func=init,
        #                     blit=True)

        playback_ratio = .25
        writer_fps = fps*playback_ratio
        FFWriter = animation.FFMpegWriter(fps=writer_fps, extra_args=['-vcodec', 'libx264'])
        anim.save(out_directory + '/trial_%03d.mp4'%plotted_trial, writer=FFWriter)
        plt.close()
        print("Video %d is done"%video_idx)

    # file_list = glob(out_directory + "/trial_*.mp4")
    # with open(out_directory + "/temp_input_kinematics_movie_file_list.txt", 'w') as f:
    #     f.writelines(["file \'%s\'"%filename for filename in file_list])

    #os.system("for f in %s/trial_*.mp4; do echo \"\'$f\'\" >> temp_input_kinematics_movie_file_list.txt; done"%out_directory)
    # os.system("ffmpeg -f concat -safe 0 -i %s/temp_input_kinematics_movie_file_list.txt -c copy %s"%(out_directory, snakemake.output[0]))
    # os.remove(out_directory + "/temp_input_kinematics_movie_file_list.txt")
    # for filename in file_list:
    #     os.remove(filename)