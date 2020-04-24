import subprocess as sp
import numpy as np

def print_commit():
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    print(commit)


def git_savefig(fig, filename):
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    fig.savefig(filename, metadata={'commit':commit})

def get_indices(input_info, trial_type):
    if trial_type == 'train':
        used_inds = input_info['trainInds'][0] - 1
    elif trial_type == 'valid':
        used_inds = input_info['validInds'][0] - 1
    elif trial_type == 'all':
        used_inds = np.sort(np.concatenate([input_info['trainInds'][0] - 1, input_info['validInds'][0] - 1]))

    return used_inds