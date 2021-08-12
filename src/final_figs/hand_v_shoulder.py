import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

if __name__=='__main__':
    output_filename = snakemake.input[0]
    fb_output_filename = snakemake.input[1]
    output = pd.read_csv(output_filename)
    fb_output = pd.read_csv(fb_output_filename)
    for o in [output, fb_output]:
        o['total_test_score']=o[['mean_test_x_score',
                                'mean_test_y_score']].mean(1)
        o['std_test_score']=o[['std_test_x_score',
                                'std_test_y_score']].mean(1)
    
    run_info = yaml.safe_load(open('../../lfads_file_locations.yml', 'r'))
    datasets = [d['name'] for d in run_info.values()]

    output['Control Type'] = 'Initial'
    fb_output['Control Type'] = 'Corrective'
    plot_columns = ['reference', 'total_test_score', 'std_test_score', 'Control Type']
    plt.figure(figsize=(12,5))
    for i, dset in enumerate(datasets):
        dset_out = output.query('dataset==@dset')[['reference','total_test_score','std_test_score', 'Control Type']]
        max_out = dset_out[dset_out.groupby('reference')['total_test_score'].transform(max)== dset_out['total_test_score']]
        max_out = max_out.iloc[:2]

        fb_dset_out = fb_output.query('dataset==@dset')[['reference','total_test_score','std_test_score', 'Control Type']]
        fb_max_out = fb_dset_out[fb_dset_out.groupby('reference')['total_test_score'].transform(max)== fb_dset_out['total_test_score']]
        fb_max_out = fb_max_out.groupby('reference').max().reset_index()

        plot_df = pd.concat([max_out[plot_columns],
                            fb_max_out[plot_columns]])
        plt.subplot(1,len(datasets),i+1)
        sns.pointplot(x='reference', y='total_test_score', hue='Control Type', data=plot_df)
        plt.errorbar([0,1], max_out.sort_values('reference')['total_test_score'].values,
                            max_out.sort_values('reference')['std_test_score'].values)
        plt.errorbar([0,1], fb_max_out.sort_values('reference')['total_test_score'].values,
                            fb_max_out.sort_values('reference')['std_test_score'].values)
        plt.title(dset)
    
    plt.savefig(snakemake.output[0])
        