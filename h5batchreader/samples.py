from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import os
import time
from warnings import warn
import random

import numpy as np
import h5py

from .utils import makeMask

class Samples(object):
    def __init__(self,
                 h5batchReader):
        self.h5br = h5batchReader
        if self.h5br.verbose:
            t0 = time.time()
            print("Samples: scanning through files %d h5files..." % len(self.h5br.h5files))
            sys.stdout.flush()

        sampleDtype = np.dtype([('file', np.int64),
                                ('row', np.int64)])
        allSamples = np.zeros(0, dtype=sampleDtype)
        totalSamples = 0
        maskedSamples = 0
        
        for h5Idx, h5filename in enumerate(self.h5br.h5files):
            num_rows = self.h5br.num_rows_this_file(h5filename)
            all_rows = np.arange(num_rows)
            h5 = h5py.File(h5filename, 'r')
            mask = makeMask(h5, 
                            exclude_if_negone_mask_datasets = self.h5br.exclude_if_negone_mask_datasets,
                            include_if_one_mask_datasets = self.h5br.include_if_one_mask_datasets)
            if mask is None:
                mask = np.ones((num_rows,), np.bool)
            numSamplesThisFile = np.sum(mask)
            maskedSamples += num_rows - numSamplesThisFile
            totalSamples += numSamplesThisFile
            samplesThisFile = np.zeros(numSamplesThisFile,  sampleDtype)
            samplesThisFile['file'] = h5Idx
            rowsThisFile = all_rows[mask]
            samplesThisFile['row'] = rowsThisFile
            allSamples = np.concatenate((allSamples, samplesThisFile))
        
        assert allSamples.shape == (totalSamples,)
        self.allSamples = allSamples
        self.totalSamples = totalSamples
        
        if self.h5br.verbose:
            sys.stdout.write("... scan took %.2f sec. There are %d samples. %d samples were masked.\n" % 
                             (time.time()-t0, self.totalSamples, maskedSamples))
            sys.stdout.flush()

        self.shuffle()
            
    def shuffle(self):
        perm=np.arange(self.totalSamples)
        np.random.shuffle(perm)
        self.allSamples = self.allSamples[perm]
        if self.h5br.verbose:
            sys.stdout.write("shuffled samples.\n")
            sys.stdout.flush()

    def _get_sub_samples(self, idxA, idxB, name):
        if idxB-idxA<=0:
            samples=None
            warn("Samples.split - %s is empty" % name)
        else:
            samples = SubSamples(h5batchReader=self.h5br,
                                 allSamples=self.allSamples[idxA:idxB],
                                 name=name)
        return samples

    def split(self, train, validation, test):
        assert train+validation+test==100, "split: train+validate+test must add up to 100, can set some to 0"
        idxA=0
        idxB=int(self.totalSamples * (float(train)/100.0))
        idxB=max(0,idxB)
        idxB=min(self.totalSamples,idxB)
        train_samples = self._get_sub_samples(idxA, idxB, 'train')
        
        idxA=idxB
        idxB=idxA + int(self.totalSamples * (float(validation)/100.0))
        validation_samples = self._get_sub_samples(idxA, idxB, 'validation')

        idxA=idxB
        idxB=idxA + int(self.totalSamples * (float(test)/100.0))
        test_samples = self._get_sub_samples(idxA, idxB, 'test')

        return train_samples, validation_samples, test_samples
    
class SubSamples(Samples):
    def __init__(self, h5batchReader, allSamples, name):
        self.h5br = h5batchReader
        self.name = name
        self.allSamples = allSamples.copy()
        self.totalSamples = len(self.allSamples)
        if self.h5br.verbose:
            print("SubSamples %s with %d samples" % (self.name, self.totalSamples))
            sys.stdout.flush()

