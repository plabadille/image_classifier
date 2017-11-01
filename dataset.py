'''
    This script will prepare a dataset to be use for training of a model. 

    Usage : no direct call, use train.py or serialize_dataset_to_hdf5.py. If the dataset is big make sure you have enought RAM or SWAP available.
    -------

    Main features :
    ---------------
        * Smartly resize images from class to respect InceptionV3 normalization
        * Transform Image file to Numpy Array
        * Generate labels list
        * Shuffle dataset

    Environment installation : Please refer to the README.md file.
    --------------------------

    Licence / feedback :
    --------------------
        Please feel free to reuse, ask me question or give me tips/comments. 
        I'm not an expert and would love to have some feed back and advise.

    @author Pierre Labadille
    @date 10/26/2017
    @version 1.0
    @todo Update to InceptionV4
'''

import sys, os, time
import cv2
from collections import defaultdict
import numpy as np

import logger

MAX_DATA_TO_PROCESS = 100 #max data to process in the same time (memory management)

def prep_data_with_cv2(filenames, n):
    processed_image_count = 0
    useful_image_count = 0
    
    X = []
    for filename in filenames:
        processed_image_count += 1
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        try:
            height, width, chan = img.shape
        except:
            print("%s can't be processed, please delete it or correct it. Check filetype, color deepness and image size according to opencv documentation" % filename)
            continue
        assert chan == 3
        aspect_ratio = float(max((height, width))) / min((height, width))
        if aspect_ratio > 2:
            continue
        # We pick the largest center square.
        centery = height // 2
        centerx = width // 2
        try:
            radius = min((centerx, centery))
            img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
            img = cv2.resize(img, (n, n), interpolation=cv2.INTER_CUBIC)
        except Exception as error:
            print("error processing %s, skipping image : %s" % (filename, repr(error)))
            continue

        X.append(img.T.transpose(1,2,0))            
        useful_image_count += 1

    return X, processed_image_count, useful_image_count

def dataset(base_dir, n):
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        #We exclude files in the base directory
        if len(files) == 1 and os.path.isfile(os.path.join(base_dir, files[0])):
            continue
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            #needed to also work on windows
            if '/' in suffix:
                suffix = suffix.lstrip("/")
                label = suffix.split("/")[0]
            else:
                suffix = suffix.lstrip("\\")
                label = suffix.split("\\")[0]
            
            d[label].append(file_path)

    tags = sorted(d.keys())

    logger.log("Processing classes data", 1)
    classes_start = time.time()
    
    processed_image_count = 0
    useful_image_count = 0
    y_tmp = []
    x = []
    for class_index, class_name in enumerate(tags):
        logger.log("Processing data from class %s" % class_name, 2)
        class_start = time.time()
        
        filenames = d[class_name]

        for i in range(0, len(filenames), MAX_DATA_TO_PROCESS):
            x_tmp, processed_image_count_tmp, useful_image_count_tmp = prep_data_with_cv2(filenames[i:i+MAX_DATA_TO_PROCESS], n)

            x += x_tmp
            x_tmp = None #we let the garbage collector free the memory from it

            for _ in range(useful_image_count_tmp):
                y_tmp.append(class_index)

            processed_image_count += processed_image_count_tmp
            useful_image_count += useful_image_count_tmp

        logger.execution_time(class_start, "Processing data from class %s" % class_name, 2)

    logger.log("Transforming X and y to numpy array", 2)
    numpy_start = time.time()

    X = np.ndarray((useful_image_count, n, n, 3), dtype=np.uint8)
    for i in range(len(x)): #memory optimization
        X[i] = x[i]
        x[i] = None

    y = np.array(y_tmp)
    logger.execution_time(numpy_start, "Transforming X and y to numpy array", 2)

    logger.log("X and Y permutation and Y to numpy", 2)
    perm_start = time.time()
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    logger.execution_time(perm_start, "X and Y permutation and Y to numpy", 2)

    logger.execution_time(classes_start, "Processing classes data", 1)

    return X, y, tags
