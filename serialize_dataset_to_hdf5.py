# @see https://gist.github.com/jfsantos/e2ef822c744357a4ed16ec0c885100a3
# @see http://docs.h5py.org/en/latest/high/dataset.html#creating-datasets

import sys, os
import h5py
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
