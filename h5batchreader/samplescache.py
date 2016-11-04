from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pdb

class SamplesCache(object):
    def __init__(self, samples, features, labels):
        '''maintain cache of samples.

        ARGS
        sample
        '''
        self._features = features
        self._labels = labels

        self._samples_dict = {}
        assert len(samples.shape)==2
        assert samples.shape[1]==3
        idx = 0
        for file_index, row, label in samples:
            self._samples_dict[(file_index,row)] = (idx, label)
            idx += 1
    
    def get_samples(self, samples=None):
        if samples is None:
            return self._features, self._labels

        assert len(samples.shape)==2
        assert samples.shape[1]==3

        as_set = set([tuple([a,b,c]) for a,b,c in samples])
        assert len(as_set)==len(samples)

        feat_shape = list(self._features.shape)
        feat_shape[0] = len(samples)
        labels_shape = list(self._labels.shape)
        labels_shape[0] = len(samples)
        features = np.zeros(tuple(feat_shape), dtype=self._features.dtype)
        labels = np.zeros(tuple(labels_shape), dtype=self._labels.dtype)
        store_index = 0
        all_cache_idx = []
        for file_index, row, label in samples:
            cache_index, stored_label = self._samples_dict[(file_index,row)]
            assert label == stored_label
            if len(features.shape)> 1:
                features[store_index,:] = self._features[cache_index,:]
            else:
                features[store_index] = self._features[cache_index]
            if len(labels.shape)>1:
                labels[store_index,:] = self._labels[cache_index,:]
            else:
                labels[store_index] = self._labels[cache_index]
            all_cache_idx.append(cache_index)
            store_index += 1
        assert len(set(all_cache_idx))==len(samples)
        return features, labels
