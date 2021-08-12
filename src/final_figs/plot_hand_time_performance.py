import pandas as pd
import h5py
import matplotlib.pyplot as plt
from ast import literal_eval

def plot_window_perf(output, color='b'):
    output['total_test_score'] = output[['mean_test_x_score','mean_test_y_score']].mean(1)
    perf_df = output[['dataset','hand_time','total_test_score']].groupby(['dataset','hand_time']).max().groupby(['hand_time']).mean()
    hand_times = []
    performance_list = []
    for hand_time, row in perf_df.iterrows():
        hand_times.append(hand_time*1000)
        performance_list.append(row['total_test_score'])

    plt.plot(hand_times, performance_list, color=color)
    plt.ylabel('Decoder Performance (r^2)')
    plt.xlabel('Time of Hand Position Relative to Movement Start (ms)')

if __name__=='__main__':
    initial_filename = '../../data/peaks/gini_firstmove_window_search.csv'
    correction_filename = '../../data/peaks/gini_correction_window_search.csv'
    initial_output = pd.read_csv(initial_filename)
    correction_output = pd.read_csv(correction_filename)

    plt.figure(figsize=(8,6))
    plot_window_perf(initial_output, color='b')
    plot_window_perf(correction_output, color='r')
    plt.legend(['Initial Movement', 'Correction'])

    plt.savefig('../../figures/final_figures/hand_time.svg')