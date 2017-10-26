'''
    This script will reduce the dataset in data_dir by the wanted numbers of features per class. It will not erase the data_dir dataset.

    Usage : $python reduce_dataset.py <new data directory name> <sample per class wanted>
    -------

    Environment installation : Please refer to the README.md file.
    --------------------------

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
from shutil import copyfile

supplied_args = sys.argv[1:]

DATA_DIRECTORY = "data_dir"
NEW_DATA_DIRECTORY = supplied_args[0] if supplied_args else sys.exit("You need to supplied a new data directory name : $python reduce_dataset.py <new data directory name> <max sample by class>")
MAX_SAMPLE_BY_CLASS = int(supplied_args[1]) if len(supplied_args) > 1 else sys.exit("You need to supplied the max sample number by class you want : $python reduce_dataset.py <new data directory name> <max sample by class>")

if not os.path.exists(DATA_DIRECTORY):
    sys.exist("The default data directory %s doesn't exist, please create it and store in it your default dataset." % DATA_DIRECTORY)

if not os.path.exists(NEW_DATA_DIRECTORY):
    os.makedirs(NEW_DATA_DIRECTORY)
classes = {}

directories_it = os.scandir(DATA_DIRECTORY)
for entry in directories_it:     
    if entry.is_file():
        continue
    
    subdir_new_path = os.path.join(NEW_DATA_DIRECTORY, entry.name)
    subdir_path = os.path.join(DATA_DIRECTORY, entry.name)

    if not os.path.exists(subdir_new_path):
        os.makedirs(subdir_new_path)
    
    classes[entry.name] = 0

    files_it = os.scandir(subdir_path)    
    for file in files_it:
        if MAX_SAMPLE_BY_CLASS > classes[entry.name]:
            file_src = os.path.join(subdir_path, file.name)
            file_dest = os.path.join(subdir_new_path, file.name)
            copyfile(file_src, file_dest)
            classes[entry.name] += 1
        else:
            break

classes_count = len(classes)
print( "%d classes with a maximum of %d samples were successfuly copied from %s directory to %s directory." % (classes_count, MAX_SAMPLE_BY_CLASS, DATA_DIRECTORY, NEW_DATA_DIRECTORY))
