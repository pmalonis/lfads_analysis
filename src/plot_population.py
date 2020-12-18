# %%
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from utils import git_savefig, get_indices
from matplotlib.backends.backend_pdf import PdfPages
import subprocess as sp
import pandas as pd

data_filename = "../data/intermediate/rockstar.p"

df = pd.read_pickle(data_filename)
ntrials = df.index[-1][0] + 1
for i in range(ntrials):
    pop_fr = df.loc[i].neural.rolling(60,center=True,win_type='gaussian').mean(std=20).sum(axis=1)*1000 
    targets = df.loc[i].kinematic.query('hit_target').index.values
    t = pop_fr.index.values
    fig = plt.figure()
    plt.plot(t, pop_fr)
    plt.vlines(targets, *fig.axes[0].get_ylim())
    plt.xlabel("Time (s)")
    plt.ylabel("Population rate")
    plt.title("trial %03d"%i)
    if i > 50:
        break