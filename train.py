'''
    This script will fine-tuned an InceptionV3 model with your own data locate in the data_dir folder (or from a name supplied folder).

    Usage : $python train.py <data_directory : optional>
    -------

    Main features :
    ---------------
        * Fine-tuned InceptionV3
        * Log and multi-save model/weight systems
        * Process raw dataset

    Recommendations :
    -----------------
        * If you have a big dataset, I recommend you to use train_with_hdf5.py, it will be slower but will let you customize the data loaded and then monitoring your RAM. It use the same features.
        * To be correctly processed : your data have to be in the data_dir folder. Then the images of each class should be put inside a folder representing the class name. You can use custom data folder if you supplied it to the script.
        * If you want to do quick run but with many class (for test purpose) you can use the reduce_dataset.py which will copy a supplied dataset and reduce it to the wanted sample per class wanted.

    Environment installation : Please refer to the README.md file.
    --------------------------

    Licence / feedback :
    -------------
        Please feel free to reuse, ask me question or give me tips/comments. 
        I'm not an expert and would love to have some feed back and advise.

    @author Pierre Labadille
    @date 10/26/2017
    @version 1.1
    @todo Add Custom Generator to load batch with Sequence
    @todo Add back data augmentation
'''

import sys, os
import datetime, time

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras import metrics
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import dataset
import net
import logger

supplied_args = sys.argv[1:]


###################
#### Constants ####
###################

## Dataset const
N = 224 #height/width for the images : InceptionV3 model require 224
CHANNELS = 3

## Training const
BATCH_SIZE = 32 #16
EPOCHS = 15
EPOCHS_FIRST = 7
BIG_EPOCHS = 3
EARLY_SOPPING_PATIENCE = 2

## Filesystem const
STARTED_DATETIME = datetime.datetime.now()
DATA_DIRECTORY = supplied_args[0] if supplied_args else "data_dir"
MODEL_PREFIX = "image-classifier_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_FOLDER_PATH = "save/" + MODEL_PREFIX + "/"
if not os.path.exists(MODEL_FOLDER_PATH):
    os.makedirs(MODEL_FOLDER_PATH)
MODEL_FILE_FULL_PATH = MODEL_FOLDER_PATH + MODEL_PREFIX


###################
#### Functions ####
###################

## Callback for logging model trends by epoch
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
    # We record model trends for each epochs using our custom callback
    model_trends = modelTrends()

    # Datagen contain a data augmentation generator defined below on the script
    model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FIRST,
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping, model_checkpoint, model_trends]
    )

    # We reload the best epoch weight before keep going
    model.load_weights(filepath)


################
#### Script ####
################

logger.header()

#1# Dataset gathering and formating
logger.log("Dataset gathering and formating", 0)
dataset_start = time.time()

data, y, tags = dataset.dataset(DATA_DIRECTORY, N)

classes_count = len(tags)
sample_count = len(y)
TRAIN_INDEX_STOP = int(0.7 * sample_count)
TEST_INDEX_START = int(sample_count - TRAIN_INDEX_STOP)

labels =  np_utils.to_categorical(y, classes_count)

logger.execution_time(dataset_start ,"Dataset gathering and formating", 0)

#2# Original InceptionV3 model loading
model = net.build_model(classes_count)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
logger.log("The Original InceptionV3 model has been loaded", 0)

#3# Train the model on the new data for a few epochs and save
logger.log("Model first train, evaluation and save", 0)
first_train_start = time.time()

filepath = MODEL_FILE_FULL_PATH + "_0.h5"

train_model(model, data[:TRAIN_INDEX_STOP], labels[:TRAIN_INDEX_STOP], data[TEST_INDEX_START:], labels[TEST_INDEX_START:], filepath)

net.save(model, tags, MODEL_FILE_FULL_PATH + "_0")
logger.execution_time(first_train_start ,"Model first train, evaluation and save", 0)

#4# Fine-tuning convolutional layers from Inception V3
logger.log("Fine-tuning convolutional layers from Inception V3", 0)
finetuning_start = time.time()

## We freeze the first 172 layers and unfreeze the top 2 inception blocks
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True
logger.log("The first 172 layers are now freezed and the top 2 inception blocks are trainable", 1)

## We recompile the model to take our modifications
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]) 
logger.log("Model has been recompiled to take our modifications", 1)

## We do the real training
logger.log("Second training", 1)
second_train_start = time.time()

# A big epoch is here a full training session (like we did on the first training bellow)
for i in range(1,BIG_EPOCHS+1):
    logger.log("Mega-epoch " + str(i), 2)
    big_epoch_start = time.time()
    
    sufix = "_" + str(i)
    filepath = MODEL_FILE_FULL_PATH + sufix + ".h5"

    # Custom train function defined upper using data-augmentation, early-stopping and model checkpoints
    train_model(model, data[:TRAIN_INDEX_STOP], labels[:TRAIN_INDEX_STOP], data[TEST_INDEX_START:], labels[TEST_INDEX_START:], filepath)

    # We save the best model for each Mega-Epoch 
    net.save(model, tags, MODEL_FILE_FULL_PATH + sufix)
    logger.execution_time(big_epoch_start, "Mega-epoch " + str(i), 2)

logger.execution_time(second_train_start, "Second training", 1)

logger.execution_time(finetuning_start ,"Fine-tuning convolutional layers from Inception V3", 0)
logger.footer(STARTED_DATETIME)
