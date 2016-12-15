
.. _tutorial:

Tutorial
===========

We'll look at how we might use the batch reader to work with
some of the psmlearn files. First::

   import glob
   import h5batchreader as hbr
   h5files = glob.glob('/scratch/davidsch/psmlearn/xtcav/amo86815_full/hdf5/amo86815_mlearn-r0*.h5')

   br = hbr.H5BatchReader(h5files,
           dsets=['xtcavimg', 'acq.enPeaksLabel'],
           exclude_if_negone_mask_datasets=['acq.enPeaksLabel'], verbose=True)
         

Each of the h5files contains the two above datasets.
If acq.enPeaksLabel is -1, we don't want to get those samples::

  br.split(train=80, validation=10, test=10)
  train_iter = br.train_iter(batchsize=4, epochs=1)

  for batch in train_iter:
     ...
     
If epochs is not given, you will continually go through the data.

Batch is a dict that looks like::

  {'batch': 0,
   'dset_groups': {},
   'dsets': {'acq.enPeaksLabel': array([1, 2, 0, 2], dtype=int8),
    'xtcavimg': array([[[ -4,  25,   2, ...,  -1,   0,  -4],
          ..., 
          [ -3, -13,  -3, ...,  11,  14,  15]]], dtype=int16)},
   'epoch': 0,
   'filesRows': array([(87, 74), (91, 129), (120, 103), (169, 212)], 
         dtype=[('file', '<i8'), ('row', '<i8')]),
   'readtime': 0.008876800537109375,
   'size': 4,
   'step': 0}

The filesRows tells you which files, and rows comprise the batch. The files are indexed.
The complete list of files is in::

  br.h5files  # a list of filenames

Dset Groups
------------

If the h5files contain many datasets of a basic type (int or float) that you want to group
together into one feature vector, you can do that::

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

   
 
