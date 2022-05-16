import numpy as np
import h5py

g = h5py.File('lfads_rockstar.h5')

outfile=h5py.File('outfile.h5')

train_data=tr[:,:300,:].transpose((2,0,1)).reshape(100,570*3,100).transpose((1,2,0))

tri=(np.repeat(g['train_inds'][:]*3,3)+np.tile(np.arange(3),g['train_inds'].shape[1]))
vi=(np.repeat(g['valid_inds'][:]*3,3)+np.tile(np.arange(3),g['valid_inds'].shape[1]))
outfile.create_dataset('train_inds',data=tri)
outfile.create_dataset('valid_inds',data=vi)