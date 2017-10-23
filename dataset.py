import sys, os, time
import cv2
from collections import defaultdict
import numpy as np

import logger

def prep_data_with_cv2(filenames, n):
    processed_image_count = 0
    useful_image_count = 0
    
    logger.log("Class images formating", 4)
    images_start = time.time()

    images = []
    for filename in filenames:
        processed_image_count += 1
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        try:
            height, width, chan = img.shape
        except:
            print(filename + "can't be processed, please delete it or correct it. Check filetype, color deepness and image size according to opencv documentation")
            continue
        assert chan == 3
        aspect_ratio = float(max((height, width))) / min((height, width))
        if aspect_ratio > 2:
            continue
        # We pick the largest center square.
        centery = height // 2
        centerx = width // 2
        radius = min((centerx, centery))
        img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
        img = cv2.resize(img, (n, n), interpolation=cv2.INTER_CUBIC)
        
        images.append(img)            
        useful_image_count += 1

    logger.execution_time(images_start, "Class images formating", 4)
    logger.log("Images transpose chan and to numpy", 4)
    transpose_start = time.time()

    X = np.ndarray((useful_image_count, n, n, 3), dtype=np.uint8)  
    for i, image in enumerate(images):
        X[i] = image.T.transpose(1, 2, 0)

    logger.execution_time(images_start, "Images transpose chan and to numpy", 4)

    return X, processed_image_count, useful_image_count

def dataset(base_dir, n):
    logger.log("Retrieving file object and class name from directory", 1)
    dataset_start = time.time()

    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
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

    logger.execution_time(dataset_start, "Retrieving file object and class name from directory", 1)
    logger.log("Processing classes data", 1)
    classes_start = time.time()
    
    processed_image_count = 0
    useful_image_count = 0
    y_tmp = []
    for class_index, class_name in enumerate(tags):
        logger.log("Processing data from class " + class_name, 2)
        class_start = time.time()
        
        filenames = d[class_name]
        
        logger.log("Formating data" + class_name, 3)

        X_tmp, processed_image_count_tmp, useful_image_count_tmp = prep_data_with_cv2(filenames, n)

        logger.execution_time(classes_start, "Formating data", 3)
        numpy_start = time.time()
        logger.log("Appending numpy class images to X", 3)

        if processed_image_count == 0:
            X = X_tmp
        else:
            X = np.concatenate((X, X_tmp))

        X_tmp = None #we let the garbage collector free the memory from it

        for i in range(0, useful_image_count_tmp):
            y_tmp.append(class_index)

        logger.execution_time(numpy_start, "Appending numpy class images to X", 3)
        logger.execution_time(class_start, "Processing data from class " + class_name, 2)

        processed_image_count += processed_image_count_tmp
        useful_image_count += useful_image_count_tmp

    logger.log("X and Y permutation and Y to numpy", 2)
    perm_start = time.time()

    y = np.array(y_tmp)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    logger.execution_time(perm_start, "X and Y permutation and Y to numpy", 2)
    logger.execution_time(classes_start, "Processing classes data", 1)

    return X, y, tags
