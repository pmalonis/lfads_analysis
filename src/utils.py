import subprocess as sp

def print_commit():
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    print(commit)


def git_savefig(fig, filename):
    '''saves matplotlib figure with hash of current git commit as metadata'''
    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    fig.savefig(filename, metadata={'commit':commit})
