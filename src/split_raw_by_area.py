from scipy.io import loadmat, savemat
import os

def get_areas(mat_data):
    '''returns list of strings representing the area name for each unit, sorted by channel name'''

if __name__=='__main__':
    
    filename = 'data/raw/raju.mat'

    data = loadmat(filename)
    chans = [k for k in data.keys() if 'Chan' in k]
    chans.sort()
    areas = ['PMd' if int(c[4:7]) in data['PMdchans'] else 'M1' for c in chans]

    area_dicts = {area:{chan:data[chan] for i,chan in enumerate(chans) if areas[i]==area} for area in set(areas)}

    for area in area_dicts.keys():
        for k in data.keys():
            if 'Chan' not in k:
                area_dicts[area][k] = data[k]

        output_filename = filename.split('.')[0] + '_%s.mat'%area
        savemat(output_filename, area_dicts[area])

    