'''
    This script will serialize a dataset (and labels) to a .HDF5 file. It's purpose is to be used before using the train_with_hdf5.py script.
    There is two great advantage to used hdf5 to store your dataset :
        * You just need to process your dataset once.
        * You can choose how much data you want to load in memory (all or just part of it) which is usefull for large dataset.

    Usage : $python serialized_dataset_to_hdf5 <dataset_to_serialized_directory> <HDF5 file name>
    -------

    Licence / feedback :
    -------------
        Please feel free to reuse, ask me question or give me tips/comments. 
        I'm not an expert and would love to have some feed back and advise.

    @author Pierre Labadille
    @date 10/26/2017
    @version 1.0
    @todo Update to InceptionV4
'''

import sys, os
import h5py # @see http://docs.h5py.org/en/latest/high/dataset.html#creating-datasets
import numpy as np

import dataset

HDF5_DIRECTORY = 'serialized_datasets'
N = 224 #height/width for the images : InceptionV3 model require 224
supplied_args = sys.argv[1:]

if not supplied_args:
    sys.exit("You need to supplied the dataset to serialized directory. Usage : $python serialized_dataset_to_hdf5 <dataset_to_serialized_directory> <HDF5 file name>")
if not len(supplied_args) > 1:
    sys.exit("You need to supplied the name of the HDF5 file. Usage : $python serialized_dataset_to_hdf5 <dataset_to_serialized_directory> <HDF5 file name>")

DATASET_DIRECTORY = supplied_args[0]

if not os.path.exists(DATASET_DIRECTORY):
    sys.exit("The dataset directory %s doesn't exist." % DATASET_DIRECTORY)

HDF5_FILENAME = supplied_args[1]
HDF5_FULL_PATH = os.path.join(HDF5_DIRECTORY, HDF5_FILENAME + ".hdf5")
LABELS_FULL_PATH = os.path.join(HDF5_DIRECTORY, HDF5_FILENAME + ".txt")

if os.path.exists(HDF5_FULL_PATH):
    sys.exit("The file %s already exist" % HDF5_FULL_PATH)
else:
    open(HDF5_FULL_PATH, 'a').close()

def serialized_dataset(X, y, tags, hdf5_full_path):
    f = h5py.File(hdf5_full_path, 'w')

    # Creating dataset to store features
    X_dset = f.create_dataset('my_data', data=X)
    # Creating dataset to store labels
    y_dset = f.create_dataset('my_labels', data=y)

    #Creating text files to store tags
    with open(LABELS_FULL_PATH, 'w+') as f:
        f.write(','.join(tags))

X, y, tags = dataset.dataset(DATASET_DIRECTORY, N)
serialized_dataset(X, y, tags, HDF5_FULL_PATH)  
