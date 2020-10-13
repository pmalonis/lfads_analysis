import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

min_submovement_ms = 50

def trial_submovements(x_vel, y_vel):
    '''Calculates submovements from single trial, based on 
    speed profile
    
    x_vel: velocity in x-axis
    y_vel: velocity in y-axis'''

    speed = np.sqrt(x_vel**2, y_vel**2)
    minima = find_peaks(-speed, width=min_submovement_ms)
    
    

if __name__=='__main__':
    pass