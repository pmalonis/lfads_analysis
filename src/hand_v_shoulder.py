import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__=='__main__':
    output_filename = '../data/peaks/hand_vs_shoulder_mack_rockstar.csv'
    fb_output_filename = '../data/peaks/fb_hand_vs_shoulder_mack_rockstar.csv'
    output = pd.read_csv(output_filename)
    fb_output = pd.read_csv(fb_output_filename)
    for o in [output, fb_output]:
        o['total_test_score']=o[['mean_test_x_score',
                                'mean_test_y_score']].mean(1)
        o['std_test_score']=o[['std_test_x_score',
                                'std_test_y_score']].mean(1)

    output['Control Type'] = 'Initial'
    fb_output['Control Type'] = 'Corrective'
    datasets = list(set(output['dataset']))
    plot_columns = ['reference', 'total_test_score', 'std_test_score', 'Control Type']
    for dset in datasets:
        dset_out = output.query('dataset==@dset')
        max_out = dset_out[dset_out.groupby('reference')['total_test_score'].transform(max)== dset_out['total_test_score']]
        
        fb_dset_out = fb_output.query('dataset==@dset')
        fb_max_out = fb_dset_out[fb_dset_out.groupby('reference')['total_test_score'].transform(max)== fb_dset_out['total_test_score']]
        
        plot_df = pd.concat([max_out[plot_columns],
                            fb_max_out[plot_columns]])
        plt.figure()
        sns.pointplot(x='reference', y='total_test_score', hue='Control Type', data=plot_df)
        # plt.errorbar([0,1], max_out['total_test_score'].values,
        #                     max_out['std_test_score'].values)
        # plt.errorbar([0,1], fb_max_out['total_test_score'].values,
        #                     fb_max_out['std_test_score'].values)
        plt.title(dset)