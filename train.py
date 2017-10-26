'''
    This script will fine-tuned an InceptionV3 model with your own data locate in the data_dir folder (or from a name supplied folder).

    Usage : $python train.py <data_directory : optional>
    -------

    Main features :
    ---------------
        * Fine-tuned InceptionV3
        * Log and multi-save model/weight systems
        * Process raw dataset
        * Cross validations to prevent overfitting
        * Data augmentation to prevent underfitting

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
    @version 1.0
    @todo Update to InceptionV4
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
N_SPLITS = 4 # the size of the test set will be 1/K (i.e. 1/n_splits), so you can tune that parameter to control the test size (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data
BATCH_SIZE = 32 #16
EPOCHS = 15
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
def train_model(model, X_train, Y_train, X_test, Y_test, filepath, datagen):
    # Early stopping if the validation loss doesn't decrease anymore
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_SOPPING_PATIENCE, verbose=1, mode='min')
    # We always keep the best model in case of early stopping
    model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)
    # We record model trends for each epochs using our custom callback
    model_trends = modelTrends()

    steps_per_epoch = len(X_train) // BATCH_SIZE if len(X_train) > BATCH_SIZE else len(X_train)
    validation_steps = len(X_test) // BATCH_SIZE if len(X_test) > BATCH_SIZE else len(X_test)

    # Datagen contain a data augmentation generator defined below on the script
    model.fit_generator(
        generator=datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE),
        validation_steps=validation_steps,
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

labels =  np_utils.to_categorical(y, classes_count)

## We use cross validation to increase the pertinence of our training and prevent it from generalizing
# @see http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

## We use a data generator to do data-augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
)

logger.execution_time(dataset_start ,"Dataset gathering and formating", 0)

#2# Original InceptionV3 model loading
model = net.build_model(classes_count)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
logger.log("The Original InceptionV3 model has been loaded", 0)

#3# Train the model on the new data for a few epochs and save
logger.log("Model first train, evaluation and save", 0)
first_train_start = time.time()

filepath = MODEL_FILE_FULL_PATH + "_0.h5"

# Cross validation loop
for i, (train_index, test_index) in enumerate(skf.split(data, y)):
    logger.log("Folds " + str(i), 2)
    # Custom train function defined upper using data-augmentation, early-stopping and model checkpoints
    train_model(model, data[train_index], labels[train_index], data[test_index], labels[test_index], filepath, datagen)

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

    # Cross validation loop
    for j, (train_index, test_index) in enumerate(skf.split(data, y)):
        logger.log("Folds " + str(j), 3)
        # Custom train function defined upper using data-augmentation, early-stopping and model checkpoints
        train_model(model, data[train_index], labels[train_index], data[test_index], labels[test_index], filepath, datagen)
    
    # We save the best model for each Mega-Epoch 
    net.save(model, tags, MODEL_FILE_FULL_PATH + sufix)
    logger.execution_time(big_epoch_start, "Mega-epoch " + str(i), 2)

logger.execution_time(second_train_start, "Second training", 1)

logger.execution_time(finetuning_start ,"Fine-tuning convolutional layers from Inception V3", 0)
logger.footer(STARTED_DATETIME)
