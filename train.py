import sys, os
import datetime, time

import numpy as np
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold

# Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
# Then we need to import it before Keras
import scipy.misc

from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras import metrics
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


import dataset
import net
import logger

np.random.seed(8585)
supplied_args = sys.argv[1:]

#-Constants
## Filesystem const
STARTED_DATETIME = datetime.datetime.now()
DATA_DIRECTORY = supplied_args[0] if supplied_args else "data_dir"
MODEL_PREFIX = "image-classifier_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_FOLDER_PATH = "save/" + MODEL_PREFIX + "/"
if not os.path.exists(MODEL_FOLDER_PATH):
    os.makedirs(MODEL_FOLDER_PATH)
MODEL_FILE_FULL_PATH = MODEL_FOLDER_PATH + MODEL_PREFIX
## Dataset const
N = 224 #height/width for the images : InceptionV3 model require 224
CHANNELS = 3
## Training const
N_SPLITS = 4 # the size of the test set will be 1/K (i.e. 1/n_splits), so you can tune that parameter to control the test size (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data
BATCH_SIZE = 32 #16
EPOCHS = 15
BIG_EPOCHS = 3
EARLY_SOPPING_PATIENCE = 2

#### Function ####

## Callback for logging model results
class modelTrends(Callback):
    def on_epoch_end(self, epoch, logs={}):
        log = "Epoch %s : Loss = %f, Val_loss = %f | Acc = %f, Val_acc = %f | Top-5-acc : %f, Val_top-5-acc : %f" % (epoch, logs['loss'], logs['val_loss'], logs['categorical_accuracy'], logs['val_categorical_accuracy'], logs['top_k_categorical_accuracy'], logs['val_top_k_categorical_accuracy'])
        logger.log(log, 3)

## Training function used by our cross_validation process.
def train_model(model, X_train, Y_train, X_test, Y_test, filepath):
    # Early stopping if the validation loss doesn't decrease anymore
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_SOPPING_PATIENCE, verbose=1, mode='min')
    # We always keep the best model in case of early stopping
    model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)
    # We record model trends for each epochs
    model_trends = modelTrends()

    model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping, model_checkpoint, model_trends]
    )

    model.load_weights(filepath)

#### Script ####

logger.header()

#-Dataset gathering and formating
logger.log("Dataset gathering and formating", 0)
dataset_start = time.time()

data, y, tags = dataset.dataset(DATA_DIRECTORY, N)

classes_count = len(tags)
sample_count = len(y)

labels =  np_utils.to_categorical(y, classes_count)

# We use cross validation to increase the pertinence of our training and prevent it from generalizing
# @see http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True) 

X = None # Performances cleanup (the value store in X are hudge)
logger.execution_time(dataset_start ,"Dataset gathering and formating", 0)

#- Original InceptionV3 model loading
logger.log("Original InceptionV3 model loading", 0)
inception_start = time.time()

model = net.build_model(classes_count)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])

logger.execution_time(inception_start ,"Original InceptionV3 model loading", 0)

#- Train the model on the new data for a few epochs and save
logger.log("Model first train, evaluation and save", 0)
first_train_start = time.time()

filepath = MODEL_FILE_FULL_PATH + "_0.h5"
#We use cross validation to train
for i, (train_index, test_index) in enumerate(skf.split(data, y)):
    logger.log("Folds " + str(i), 2)
    train_model(model, data[train_index], labels[train_index], data[test_index], labels[test_index], filepath)
net.save(model, tags, MODEL_FILE_FULL_PATH + "_0")

logger.execution_time(first_train_start ,"Model first train, evaluation and save", 0)

#- Start fine-tuning convolutional layers from Inception V3
logger.log("Fine-tuning convolutional layers from Inception V3", 0)
finetuning_start = time.time()
## We freeze the first 172 layers and unfreeze the top 2 inception blocks
logger.log("Freezing the first 172 layers and unfreeze the top 2 inception blocks", 1)
freeze_start = time.time()

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

logger.execution_time(freeze_start, "Freezing the first 172 layers and unfreeze the top 2 inception blocks", 1)

## We recompile the model to take our modifications
logger.log("Recompiling the model to take our modifications", 1)
compile_start = time.time()

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]) # we use SGD with a low learning rate

logger.execution_time(compile_start, "Recompiling the model to take our modifications", 1)

## We do the training
logger.log("Second training", 1)
second_train_start = time.time()

for i in range(1,BIG_EPOCHS+1):
    print("mega-epoch %d/%d" % (i,BIG_EPOCHS))
    logger.log("Mega-epoch " + str(i), 2)
    big_epoch_start = time.time()
    
    sufix = "_" + str(i)
    filepath = MODEL_FILE_FULL_PATH + sufix + ".h5"

    #we use cross validation to train
    for i, (train_index, test_index) in enumerate(skf.split(data, y)):
        logger.log("Folds " + str(i), 3)
        train_model(model, data[train_index], labels[train_index], data[test_index], labels[test_index], filepath)
    net.save(model, tags, MODEL_FILE_FULL_PATH + sufix)
    
    logger.execution_time(big_epoch_start, "Mega-epoch " + str(i), 2)

logger.execution_time(second_train_start, "Second training", 1)

logger.execution_time(finetuning_start ,"Fine-tuning convolutional layers from Inception V3", 0)

logger.footer(STARTED_DATETIME)
