'''
    This script will output the 5-top predictions per sample in the folder data_to_predict using the supplied trained model. 

    Usage : $python predict.py <path_to_model without ext>
    -------

    Main features :
    ---------------
        * Perform image prediction on the model
        * Retrieve only the top-5 predictions from each image and class name.

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

import os, sys, json

import operator
import cv2
import numpy as np

from keras.models import Sequential, model_from_json

supplied_args = sys.argv[1:]
if not supplied_args:
    sys.exit("You need to supply the path to the model version wanted (please exclude extension). The json and h5 files need to have the same name. Usage : $python predict.py <path_to_model without ext>")

MODEL_WEIGHT_PATH = supplied_args[0] + '.h5'
MODEL_JSON_PATH = supplied_args[0] + '.json'
MODEL_LABELS_PATH = supplied_args[0] + '-labels.json'
PREDICT_DIR_PATH = 'data_to_predict'

N = 224 #height/width for the images : InceptionV3 model require 224
CHANNELS = 3

# We reload the wanted model
with open(MODEL_JSON_PATH, 'r') as f:
    loaded_model_json = f.read()
model = model_from_json(loaded_model_json)

# We retrieve the labels
with open(MODEL_LABELS_PATH, 'r') as f:
    labels_json = f.read()
labels = json.loads(labels_json)
# We reload the needed weight
model.load_weights(MODEL_WEIGHT_PATH)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, N, N, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        img = cv2.imread(image_file, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
        image = cv2.resize(img, (N, N), interpolation=cv2.INTER_CUBIC)

        data[i] = image.T.transpose(1, 2, 0)
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data, count

def get_n_max_value(array, n):
    max_array = []

    for i in range(n):
        max_index, max_value = max(enumerate(array), key=operator.itemgetter(1))
        max_array.append((max_index, max_value))
        if max_index != None:
            array[max_index] = 0.0

    return max_array

images_path = []
images_name = []
for filename in os.listdir(PREDICT_DIR_PATH):
    if filename != ".gitignore":
        images_name.append(filename)
        images_path.append(os.path.join(PREDICT_DIR_PATH, filename))

images, count = prep_data(images_path)

# We query the model to get it prediction
predictions = model.predict(images, verbose=0)

MAX_PREDICT_VALUE = 5 if len(labels) >= 5 else len(labels)
# We print them for each pictures in a human readable format.
for i, prediction in enumerate(predictions):
    filename = images_name[i]
    print("Predictions for image %s :" % filename)

    top_n_predictions = get_n_max_value(prediction.tolist(), MAX_PREDICT_VALUE)

    for label_index, prediction in top_n_predictions:
        print('* %s : %f' % (labels[label_index], prediction*100))
    print()
