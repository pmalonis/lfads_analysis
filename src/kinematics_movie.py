import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
from matplotlib import animation
from scipy import io

t_start = 361.52
t_end = 364.84
filename = '../data/raw/mack_edited.mat'
fps = 50 #frame rate of saved animation
kinematic_fs = 500 #frame rame of kinematic data
data = io.loadmat(filename)
fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlim(-250, 100)
ax.set_ylim(100, 300)
cursor_ln, = ax.plot([], [], 'b.', markersize=8)
target_ln, = ax.plot([], [], marker='s', color='r', markersize=50)
#target_ln, = ax.plot([], [], 'r.')

target_pos = ([], [])
hit_target_idx = 0
endmv_idx = 0
st_trial_idx = 0

def update_target(data, t):
    '''finds the next target in data dictionary and returns its index and position'''
    hit_target_idx = np.where(data['hit_target'] - t > 0)[0][0]
    t_next_target = data['hit_target'][hit_target_idx]
    idx = np.argmin(np.abs(t_next_target - data['x'][:,0]))
    target_pos = (data['x'][idx,1], data['y'][idx,1])
    return hit_target_idx, target_pos

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
        i = int((t - data['x'][0,0]) * kinematic_fs)
        cursor_ln.set_data(data['x'][i,1], data['y'][i,1])
    else:
        raise ValueError("t out of range of experiment time")

    # plotting target
    if np.sum(target_pos):
        target_visible = True
    else:
        target_visible = False

    if target_visible:
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
    else:
        if t > data['st_trial'][st_trial_idx]:
            hit_target_idx, target_pos = update_target(data, t)

    target_ln.set_data(*target_pos)
    return cursor_ln, target_ln

if __name__=='__main__':
    anim = animation.FuncAnimation(fig, animate, frames=np.arange(t_start, t_end, 1./fps), init_func=init, blit=True)
# anim = animation.FuncAnimation(fig, animate, frames=range(0, data['x'].shape[0], 10), init_func=init,
#                     blit=True)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    FFWriter = animation.FFMpegWriter(fps=50, extra_args=['-vcodec', 'libx264'])
    anim.save('../figures/presentation_clip.mp4', writer=FFWriter)
