'''
    This script will give some usefull methods to load the original InceptionV3 model and to save/load trained model.

    Usage : no direct call, used by train.py, train_with_hdf5.py.
    -------
    
    IMPORTANT NOTE : Please note this file is identical or nearly identical to https://github.com/danielvarga/keras-finetuning/blob/master/net.py
    ----------------

    @author Daniel Varga
    @link https://github.com/danielvarga/keras-finetuning
    @date 08/31/2016
    @todo Update to InceptionV4
'''

import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# create the base pre-trained model
def build_model(nb_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    print("starting model compile")
    compile(model)
    print("model compile done")
    return model


def save(model, tags, prefix, sufix=""):
    model.save_weights(prefix+sufix+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+sufix+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+sufix+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)


def load(prefix, sufix=""):
    # load json and create model
    with open(prefix+sufix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+sufix+".h5")
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
