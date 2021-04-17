import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from matplotlib import animation
from scipy import io
import h5py
import os
from glob import glob
from utils import get_indices

# lfads_file = 'data/model_output/rockstar_valid.h5'
# filename = 'data/raw/rockstar.mat'
# input_info_file = 'data/model_output/rockstar_inputInfo.mat'
# out_directory = '/home/pmalonis/'

fps = 50 #frame rate at which to read data
playback_ratio = 0.25  # speed of playback (1 indicates real speed)
kinematic_fs = 500 #frame rame of kinematic data

def update_target(data, t):
    '''finds the next target in data dictionary and returns its index and position'''
    hit_target_idx = np.where(data['hit_target'] - t > 0)[0][0]
    t_next_target = data['hit_target'][hit_target_idx]
    idx = np.argmin(np.abs(t_next_target - data['x'][:,0]))
    target_pos = (data['x'][idx,1], data['y'][idx,1])
    return hit_target_idx, target_pos

def update_target_pandas(trial_df, t):
    t_targets = trial_df.kinematic.query('hit_target').index.values[1:]
    time_vec = trial_df.index.values
    x = trial_df.kinematic['x'].values
    y = trial_df.kinematic['y'].values

    hit_target_idx = np.where(t_targets - t > 0)[0][0]
    t_next_target = t_targets[hit_target_idx]
    try:
        idx = np.argmin(np.abs(t_next_target - time_vec))
    except:
        import pdb;pdb.set_trace()
    target_pos = (x[idx], y[idx])
    return hit_target_idx, target_pos

#TODO make scrolling plot optional
def animate_trial_pandas(trial_df, scroll_data, scroll_dt, title='', fps=fps):
    '''Creates matplotlib animation of data for single trial. The animation
    shows the cursor position over time as a dot, and the target positions as a square. 
    in addition, a scrolling plot of any variable is also plotted
    
    Parameters
    mat_data:  data dictionary originating from raw data .mat file
    trial_index: index of trial in mat_data['cpl_st_trial_rew'] to use
    scroll_data: Data to include in scrolling plot. Each row represents a different time series
    that will be included in the scrolling plot
    variable_dt: Sampling time of the scrolled data
    title: Title to display on animation from kinematics
    '''

    trial_t = np.arange(scroll_data.shape[1]) * scroll_dt
    trial_len = scroll_data.shape[1] * scroll_dt
    t_start = 0
    t_end = t_start + trial_len
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    xmin = trial_df.kinematic['x'].min()
    xmax = trial_df.kinematic['x'].max()
    ymin = trial_df.kinematic['y'].min()
    ymax = trial_df.kinematic['y'].max()
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(title)
    cursor_ln, = ax[0].plot([], [], 'r.', markersize=8)
    target_ln, = ax[0].plot([], [], marker='s', color='b', markersize=8)

    scroll_lns = []
    for scroll_idx in range(scroll_data.shape[0]):
        ax[1].plot(trial_t, scroll_data[scroll_idx,:])


    scroll_ymin = np.min(scroll_data) #TODO base on min and max of scroll_data
    scroll_ymax = np.max(scroll_data)
    scroll_range = scroll_ymax - scroll_ymin
    scroll_ymin -= scroll_range*0.05
    scroll_ymax += scroll_range*0.05
    time_ln = ax[1].vlines(0, scroll_ymin, scroll_ymax)

    #plotting target appearance times
    t_targets = trial_df.kinematic.query('hit_target').index.values[1:]
    ax[1].vlines(t_targets, scroll_ymin, scroll_ymax, 'r')

    ax[1].set_ylim(scroll_ymin, scroll_ymax)
    ax[1].set_xlabel('Time(s)')

    global target_pos, hit_target_idx, endmv_idx, st_trial_idx
    hit_target_idx, target_pos = update_target_pandas(trial_df, t_start)

    def init():
        cursor_ln.set_data(trial_df.kinematic['x'].loc[0:].iloc[0], trial_df.kinematic['x'].loc[0:].iloc[0])
        target_ln.set_data(trial_df.kinematic['x'].loc[t_targets[0]], trial_df.kinematic['x'].loc[t_targets[0]])
        return cursor_ln, target_ln

    def animate(t):
        '''Animates kinematic frame at time t by plotting cursor position and, if t
        is a time during a trial, the target position'''

        global target_pos, hit_target_idx, endmv_idx, st_trial_idx

        # plotting cursor
        time_vec = trial_df.index.values
        x = trial_df.kinematic['x'].values
        y = trial_df.kinematic['y'].values

        if 0 <= t < np.max(time_vec):
            i = np.argmin(np.abs(t - time_vec))
            cursor_ln.set_data(x[i], y[i])
        else:
            raise ValueError("t out of range of experiment time")

        # # plotting target
        # if np.sum(target_pos):
        #     target_visible = True
        # else:
        #     target_visible = False

        # if target_visible:
        if t > t_targets[hit_target_idx]: #if target has been hit
            # sets conditions for end of trial, which depends on whether
            # the current trial is the last one
            trial_ended = t >= trial_len

            if trial_ended:
                # if trial is over, update the indices that allow finding the
                # next trial
                target_pos = ([], [])
            else:
                # if trial is not over, update the target position and the next
                # target acquisition time
                hit_target_idx, target_pos = update_target_pandas(trial_df, t)

        target_ln.set_data(*target_pos)

        #plotting
        lfads_idx = np.digitize(t-t_start, trial_t) - 1 #assumpes t_start is trial start
        time_ln.set_segments([np.array([[trial_t[lfads_idx], scroll_ymin], [trial_t[lfads_idx], scroll_ymax]])])
        for scroll_idx, scroll_ln in enumerate(scroll_lns):
            scroll_ln.set_data(trial_t[lfads_idx], scroll_data[scroll_idx,:])

        return (cursor_ln, target_ln, *scroll_lns)

    frames = np.arange(t_start, t_end, 1./fps)
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                    init_func=init, blit=True)

    return anim, frames

#TODO make scrolling plot optional
def animate_trial(mat_data, trial_index, scroll_data, scroll_dt, title='', fps=fps):
    '''Creates matplotlib animation of data for single trial. The animation
    shows the cursor position over time as a dot, and the target positions as a square. 
    in addition, a scrolling plot of any variable is also plotted
    
    Parameters
    mat_data:  data dictionary originating from raw data .mat file
    trial_index: index of trial in mat_data['cpl_st_trial_rew'] to use
    scroll_data: Data to include in scrolling plot. Each row represents a different time series
    that will be included in the scrolling plot
    variable_dt: Sampling time of the scrolled data
    title: Title to display on animation from kinematics
    '''

    trial_t = np.arange(scroll_data.shape[1]) * scroll_dt
    trial_len = scroll_data.shape[1] * scroll_dt
    t_start = mat_data['cpl_st_trial_rew'][trial_index, 0].real
    t_end = t_start + trial_len
    print(trial_len)
    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    ax[0].set_xlim(-100, 100)
    ax[0].set_ylim(100, 300)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title(title)
    cursor_ln, = ax[0].plot([], [], 'r.', markersize=8)
    target_ln, = ax[0].plot([], [], marker='s', color='b', markersize=8)

    scroll_lns = []
    for scroll_idx in range(scroll_data.shape[0]):
        ax[1].plot(trial_t, scroll_data[scroll_idx,:])


    scroll_ymin = np.min(scroll_data) #TODO base on min and max of scroll_data
    scroll_ymax = np.max(scroll_data)
    scroll_range = scroll_ymax - scroll_ymin
    scroll_ymin -= scroll_range*0.05
    scroll_ymax += scroll_range*0.05
    time_ln = ax[1].vlines(0, scroll_ymin, scroll_ymax)

    #plotting target appearance times
    t_targets = mat_data['hit_target'][(mat_data['hit_target'] >= t_start) & (mat_data['hit_target'] < t_end)]
    t_targets -= t_start
    ax[1].vlines(t_targets, scroll_ymin, scroll_ymax, 'r')

    ax[1].set_ylim(scroll_ymin, scroll_ymax)
    ax[1].set_xlabel('Time(s)')

    global target_pos, hit_target_idx, endmv_idx, st_trial_idx
    hit_target_idx, target_pos = update_target(mat_data, t_start)
    st_trial_idx = np.where(mat_data['st_trial'] - t_start > 0)[0][0]
    endmv_idx = np.where(mat_data['endmv'] - t_start > 0)[0][0]

    def init():
        cursor_ln.set_data([], [])
        target_ln.set_data([], [])
        return cursor_ln, target_ln

    def animate(t):
        '''Animates kinematic frame at time t by plotting cursor position and, if t
        is a time during a trial, the target position'''

        global target_pos, hit_target_idx, endmv_idx, st_trial_idx

        # plotting cursor
        if np.min(mat_data['x'][:,0]) < t < np.max(mat_data['x'][:,0]):
            i = np.argmin(np.abs(t - mat_data['x'][:,0]))
            cursor_ln.set_data(mat_data['x'][i,1], mat_data['y'][i,1])
        else:
            raise ValueError("t out of range of experiment time")

        # # plotting target
        # if np.sum(target_pos):
        #     target_visible = True
        # else:
        #     target_visible = False

        # if target_visible:
        if t > mat_data['hit_target'][hit_target_idx]: #if target has been hit
            # sets conditions for end of trial, which depends on whether
            # the current trial is the last one
            if st_trial_idx + 1 >= len(mat_data['st_trial']):
                trial_ended = t > mat_data['endmv'][endmv_idx]
            else:
                trial_ended = (t > mat_data['endmv'][endmv_idx]) #The time between an endmv event and the next st_trial can be smaller than the frame rate

            if trial_ended:
                # if trial is over, update the indices that allow finding the
                # next trial
                st_trial_idx = np.where(mat_data['st_trial'] - t > 0)[0][0]
                endmv_idx = np.where(mat_data['endmv'] - t > 0)[0][0]
                if t < mat_data['st_trial'][st_trial_idx]:
                    target_pos = ([], [])
                else:
                    hit_target_idx, target_pos = update_target(mat_data, t)
            else:
                # if trial is not over, update the target position and the next
                # target acquisition time
                hit_target_idx, target_pos = update_target(mat_data, t)

        target_ln.set_data(*target_pos)

        #plotting
        lfads_idx = np.digitize(t-t_start, trial_t) - 1 #assumpes t_start is trial start
        time_ln.set_segments([np.array([[trial_t[lfads_idx], scroll_ymin], [trial_t[lfads_idx], scroll_ymax]])])
        for scroll_idx, scroll_ln in enumerate(scroll_lns):
            scroll_ln.set_data(trial_t[lfads_idx], scroll_data[scroll_idx,:])


        return (cursor_ln, target_ln, *scroll_lns)

    frames = np.arange(t_start, t_end, 1./fps)
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                    init_func=init, blit=True)

    return anim, frames

if __name__=='__main__':
    lfads_file = snakemake.input[0]
    filename = snakemake.input[1]
    input_info_file = snakemake.input[2]
    out_directory = os.path.dirname(snakemake.output[0])
    os.makedirs(out_directory, exist_ok=True)

    input_info = io.loadmat(input_info_file)
    used_inds = get_indices(input_info, snakemake.wildcards.trial_type)

    with h5py.File(lfads_file) as h5file:
        dt = 0.01    
        trial_t = np.arange(h5file['controller_outputs'].shape[1]) * dt
        trial_len = trial_t[-1] + dt

        data = io.loadmat(filename)
        used_trials =  np.where(np.diff(data['cpl_st_trial_rew'].real, axis=1) > trial_len)[0]

        #TODO: replace with animate_trial function and test
        for video_idx, plotted_trial in enumerate(used_inds):
            t_start = data['cpl_st_trial_rew'][used_trials[plotted_trial],0].real
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
                ax[1].plot(trial_t, h5file['controller_outputs'][plotted_trial,:,input_idx])
                
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
                        trial_ended = t > data['endmv'][0,endmv_idx]
                    else:
                        trial_ended = (t > data['endmv'][0,endmv_idx]) #The time between an endmv event and the next st_trial can be smaller than the frame rate

                    if trial_ended:
                        # if trial is over, update the indices that allow finding the
                        # next trial
                        st_trial_idx = np.where(data['st_trial'] - t > 0)[0][0]
                        endmv_idx = np.where(data['endmv'] - t > 0)[0][0]
                        if t < data['st_trial'][0,st_trial_idx]:
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
                    input_ln.set_data(trial_t[lfads_idx], h5file['controller_outputs'][video_idx, :, input_idx])

                return (cursor_ln, target_ln, *input_lns)

            anim = animation.FuncAnimation(fig, animate, frames=np.arange(t_start, t_end, 1./fps), init_func=init, blit=True)
            # anim = animation.FuncAnimation(fig, animate, frames=range(0, data['x'].shape[0], 10), init_func=init,
            #                     blit=True)

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
