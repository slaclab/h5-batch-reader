# h5-mlearn-minibatch

package to prepare minibatches from h5 input for a machine learning framework.

The intent is to handle a collection of hdf5 files that have a relatively flat 
schema. Something like
```
   /features
   /labels
```
The class H5MiniBatchReader will read first through these files and identify
the samples and labels, and imbalances in the classes (i.e, there are more 
samples labeled 0 then labeled 1). The class provides options to restrict the
samples used to reach some balance target.

Presently, one can look at the examples in 

  https://github.com/davidslac/xtcav-mlearn-doc

namely

  https://github.com/davidslac/xtcav-mlearn-doc/blob/master/tensorflow_simple.py

for an example of use.

