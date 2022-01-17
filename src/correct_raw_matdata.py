from scipy import io
import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
cfg = yaml.safe_load(open(config_path, 'r'))
run_info = yaml.safe_load(open('../lfads_file_locations.yml', 'r'))

def correct_timing(filename, output):
    '''corrects timing error in raw files
    
    
    filename: filename of mat file to correct
    output: filename of output file'''

    matdata = io.loadmat(filename)

    if matdata.get('timing_corrected'):
        return
    else:
        neurons = [n for n in matdata.keys() if n[:4] == 'Chan']
        for neuron in neurons:
            matdata[neuron] -= cfg['neural_offset']

        matdata['timing_corrected'] = True

        io.savemat(output, matdata)

if __name__ == '__main__':
    datasets = list(cfg['datasets'].keys())
    for dataset in datasets:
        filename = os.path.basename(cfg['datasets'][dataset]['raw'])
        file_path = '../data/raw/' + filename
        correct_timing(file_path, file_path)