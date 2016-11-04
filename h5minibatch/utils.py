from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import h5py

_str2np = {'float32':np.float32,
           'float64':np.float64,
           'int8':np.int8,
           'int16':np.int16,
           'int32':np.int32,
           'int64':np.int64,
           'int8':np.int8,
           'uint16':np.uint16,
           'uint32':np.uint32,
           'uint64':np.uint64}

def str2np_dtype(dtypestr):
    global _str2np
    assert dtypestr in _str2np
    return _str2np[dtypestr]

def makeMask(h5, exclude_if_negone_mask_datasets=[], include_if_one_mask_datasets=[]):
    if isinstance(h5, str):
        h5 = h5py.File(h5,'r')
    mask = None

    for maskList, maskVal in zip([exclude_if_negone_mask_datasets,
                                  include_if_one_mask_datasets],
                                 [-1, 1]):
        for dsname in maskList:
            assert dsname in h5.keys(), "masking dataset: %s not in h5file" % dsname
            curmask = h5[dsname][:] == maskVal
            if maskVal == -1:
                curmask = np.logical_not(curmask)
            if mask is None:
                mask = curmask
            else:
                mask = np.logical_and(mask, curmask)

    return mask
        

def get_balanced_samples(samples, num_outputs, class_labels_max_imbalance_ratio, verbose=False):
    '''takes matrix with 3 columns, file_idx, sample_idx, label
    returns a matrix that is a selection of those rows, with
    numSamples rows. There should be an equal amount of each 
    label in the returned matrix
    '''
    assert class_labels_max_imbalance_ratio > 0.0
    if class_labels_max_imbalance_ratio < 1.0:
        class_labels_max_imbalance_ratio = (1.0 / class_labels_max_imbalance_ratio)
    assert class_labels_max_imbalance_ratio >= 1.0

    label2rows = {}
    min_samples_in_a_label = len(samples)
    for label in range(num_outputs):
        label2rows[label] = np.where(samples[:,2]==label)[0]
        min_samples_in_a_label = min(min_samples_in_a_label, len(label2rows[label]))

    max_labels_per_class_to_balance = int(round(min_samples_in_a_label * class_labels_max_imbalance_ratio))

    balanced_samples = np.zeros((0,3), dtype=np.int32)
    for rows in label2rows.values():
        balanced_rows = rows[0:max_labels_per_class_to_balance]
        balanced_samples = np.vstack((balanced_samples, samples[balanced_rows,:]))

    as_set = set([tuple([a,b,c]) for a,b,c in balanced_samples])
    assert len(as_set)==len(balanced_samples), "number of balanced samples=%d != unique #=%d" % \
        (len(balanced_samples),len(as_set))
    
    if verbose:
        print("After balancing per ratio %.2f, there are %d examples" % (class_labels_max_imbalance_ratio, len(balanced_samples)))
        count_in_largest_class = 0
        for label in range(num_outputs):
            num_this_label = sum(balanced_samples[:,2]==label)
            count_in_largest_class=max(count_in_largest_class,num_this_label)
            percent_this_label = 100.0 * num_this_label / len(balanced_samples)
            print("   label=%3d   %8d examples  (%5.1f%% of dataset)" % (label, num_this_label, percent_this_label))
        print("classifier that always picks label for largest class performs at: %.1f%%" % ((count_in_largest_class/len(balanced_samples))*100.0,))
    return balanced_samples


def convert_to_one_hot(labels, numLabels):
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "labels must have entries not in [0,%d)" % numLabels
    return labelsOneHot

def load_features_and_labels(dataset, 
                             samples, 
                             h5files, 
                             preprocess=None,
                             add_channel_to_2D=None,
                             one_hot_num_outputs=None):
    '''returns features, labels
    ARGS
      samples - 2D integer array with 3 columns:
    fileIdx, row, label
    fileIdx is an index into the list of files h5
    '''
    ############# helper functions
    def get_features_shape_and_dtype(ds, add_channel_to_2D):
        assert len(ds.shape)<4, "do not support 4d or higher featuresets"
        if len(ds.shape) == 1:
            features_shape = (len(samples),)
        elif len(ds.shape) == 2:
            features_shape = (len(samples), ds.shape[1])
        elif len(ds.shape)==3:
            if add_channel_to_2D == 'row_column_channel':
                features_shape = (len(samples), ds.shape[1], ds.shape[2], 1)
            elif add_channel_to_2D == 'channel_row_column':
                features_shape = (len(samples), 1, ds.shape[1], ds.shape[2])

            else:
                features_shape = (len(samples), ds.shape[1], ds.shape[2])
        elif len(ds.shape)==4:
                features_shape = (len(samples), ds.shape[1], ds.shape[2], ds.shape[3])
        return features_shape, ds.dtype

    def copy_samples(ds, read_from, features, store_at, add_channel_to_2D):
        if len(features.shape)==1:
            features[store_at] = ds[read_from]
        elif len(features.shape)==4 and len(ds.shape)==3:
            # images, where we add one channel to features
            if add_channel_to_2D == 'row_column_channel':
                features[store_at,:,:,0] = ds[read_from,:]
            elif add_channel_to_2D == 'channel_row_column':
                features[store_at,0,:,:] = ds[read_from,:]
            else:
                raise Exception("copying from 3D to 4D, but add_channel_to_2D not 'row_column_channel' or 'channel_row_column', it is %s" % add_channel_to_2D)
        else:
            features[store_at,:] = ds[read_from,:]

    ######### end helpers                                    
    assert add_channel_to_2D in [None, 'row_column_channel', 'channel_row_column'], "add_channel_to_2D is %s" % add_channel_to_2D
    if preprocess is None:
        preprocess = []
    assert isinstance(preprocess, list) or isinstance(preprocess, tuple), 'preprocess must be None, or a list of strings'
    for step in preprocess:
        assert step in ['mean', 'log', 'float32', 'float64']

    if one_hot_num_outputs:
        labels = convert_to_one_hot(samples[:,2], one_hot_num_outputs)
    else:
        labels = samples[:,2].astype(np.int32)

    features_shape, features_dtype = get_features_shape_and_dtype(h5py.File(h5files[0],'r')[dataset], 
                                                                  add_channel_to_2D)
    features = np.zeros(features_shape, dtype=features_dtype)
    
    file_idx_2_read_store = {}
    for feature_row, sample_row in enumerate(samples):
        file_idx, read_row, label = sample_row
        if file_idx not in file_idx_2_read_store:
            file_idx_2_read_store[file_idx] = {'read_from':[], 'store_at':[]}
        file_idx_2_read_store[file_idx]['read_from'].append(read_row)
        file_idx_2_read_store[file_idx]['store_at'].append(feature_row)
    
    for file_idx, read_store in file_idx_2_read_store.iteritems():
        read_from = np.array(read_store['read_from'])
        store_at = np.array(read_store['store_at'])
        argsort_read_from = np.argsort(read_from)
        read_from = np.sort(read_from)
        store_at = store_at[argsort_read_from]
        h5file = h5files[file_idx]
        h5 = h5py.File(h5file, 'r')
        copy_samples(h5[dataset], read_from, features, store_at, add_channel_to_2D)

    features = apply_preprocess_steps(features, preprocess)
    return features, labels

