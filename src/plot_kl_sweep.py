import h5py
import yaml
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

if __name__=='__main__':
    run_info = yaml.safe_load(open('../lfads_file_locations.yml'))
    output_name = '../figures/kl_sweep_spectrum.pdf'
    
    plots_per_page = 5

    with PdfPages(output_name) as pdf:
        for dataset in run_info.keys():
            dset_params = sorted(run_info[dataset]['params'].items(),
                                key=lambda d:d[1]['param_values']['kl_co_weight'])

            dset_params = [(p,d) for p,d in dset_params if d['param_values']['ar_prior_dist']=='laplace']
            for i, (param_str, params) in enumerate(dset_params):
                lfads_filename = '../data/model_output/' + '_'.join([dataset, param_str, 'all.h5'])
                kl_weight = params['param_values']['kl_co_weight']
                with h5py.File(lfads_filename, 'r') as h5file:
                    co = h5file['controller_outputs'][:]

                if kl_weight > 2.0:
                    pdf.savefig(fig)
                    fig = plt.figure()
                
                if i%plots_per_page == 0:
                    if i > 0:
                        pdf.savefig(fig)
                    fig = plt.figure()

                plt.subplot(plots_per_page, 1, i%plots_per_page + 1)
                plt.plot(np.arange(co.shape[1])*10, co[0,:,:],linewidth=0.5)
                plt.ylim([-1,1])
                plt.title('%s  (%s)'%(kl_weight,dataset))
    
    plt.close('all')