from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########## imports ##########
# python system
import os
from warnings import warn

#3rd party

# this package
from h5minibatch import DataSetGroup

###########
FNAME = '/reg/d/ana01/temp/davidsch/ImgMLearnSmall/amo86815_mlearn-r070-c0000.h5'

def test_datasetGroup():
    fvecnames=['bld.ebeam.ebeamCharge',
               'bld.ebeam.ebeamDumpCharge',
               'bld.ebeam.ebeamEnergyBC1']
    dgroup = DataSetGroup(name='bld', dsets=fvecnames)
    print(dgroup)
    print("%r" % dgroup)
    if os.path.exists(FNAME):
        dgroup.validate(FNAME)
    else:
        warn("not testing validate, file %s doesn't exist" % FNAME)
        
if __name__ == '__main__':
    test_datasetGroup()
    
