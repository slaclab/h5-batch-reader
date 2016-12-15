import glob
import h5batchreader as hbr
h5files = glob.glob('/scratch/davidsch/psmlearn/xtcav/amo86815_full/hdf5/amo86815_mlearn-r0*.h5')
assert len(h5files)>0

br = hbr.H5BatchReader(h5files,
        dsets=['xtcavimg', 'acq.enPeaksLabel'],
        exclude_if_negone_mask_datasets=['acq.enPeaksLabel'], verbose=True)


br.split(train=80, validation=10, test=10)
train_iter = br.train_iter(batchsize=8, epochs=1)

for batch in train_iter:
    print batch
    break

fvec = hbr.DataSetGroup(name='fvec',
                        dsets=['bld.ebeam.ebeamL3Energy',
                               'bld.gasdet.f_11_ENRC'])
br = hbr.H5BatchReader(h5files,
                       dsets=['xtcavimg', 'acq.enPeaksLabel'],
                       dset_groups=[fvec],      
                       exclude_if_negone_mask_datasets=['acq.enPeaksLabel'], verbose=True)

br.split()
train_iter = br.train_iter(batchsize=4)

for batch in train_iter:
    print batch
    break
