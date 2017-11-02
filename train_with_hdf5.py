'''
    This script will fine-tuned an InceptionV3 model with your own data serialized in hdf5 file.

    Usage : $python train_with_hdf5.py <hdf5_filename>
    -------

    Main features :
    ---------------
        * Fine-tuned InceptionV3
        * Log and multi-save model/weight systems
        * Only load segmented part of your dataset (you choose it) at the same time to avoid ram issue on large dataset.
        * Cross validations to prevent overfitting
        * Data augmentation to prevent underfitting

    Recommendations :
    -----------------
        * If you have a small dataset (or a war machin with a lot of ram), I recommend you to use train.py, it will be quicker
    and use the same feature (except the hability to train from hdf5).
        * If you already have a hdf5 file you may have to do some modification in the load_data() function.
        * If you don't have a hdf5 file of your dataset, I strongly recommend to use the script serialize_dataset_to_hdf5.py. It'll make the files you need to run this script (make sure the dependencies of dataset.py are installed).
        * Choose carefuly the MAX_DATA_LOAD const. For a 5Go dataset (101 classes / 1k pict by class) I used 21Go to load 30K max data. This is mainly due to data-augmentation which duplicate your dataset in memory.

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
import h5py
import math, random

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras import metrics
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

import net
import logger

supplied_args = sys.argv[1:]
if not supplied_args:
    sys.exit("You need to supply the HDF5 (dataset) TXT (tags) filename (without extension but have to be .hdf5 and .txt) where the dataset is stored. Usage : $python train_with_hdf5.py <hdf5_filename>")


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
MAX_DATA_LOAD = 30000 #max data to load in memory at the same time

## Filesystem const
HDF5_FOLDER = "serialized_datasets"
MODEL_PREFIX = "image-classifier_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_FOLDER_PATH = "save/" + MODEL_PREFIX + "/"

## Computed or script supplied const
STARTED_DATETIME = datetime.datetime.now()
HDF5_FILENAME = supplied_args[0]
HDF5_FULL_PATH = os.path.join(HDF5_FOLDER, HDF5_FILENAME + ".hdf5")
TAGS_FULL_PATH = os.path.join(HDF5_FOLDER, HDF5_FILENAME + ".txt")
MODEL_FILE_FULL_PATH = MODEL_FOLDER_PATH + MODEL_PREFIX

## We exit if supplied args are not correct
if not os.path.exists(HDF5_FULL_PATH):
    sys.exit("The file %s doesn't exist. Please generate it using the serialize_dataset_to_hdf5.py file or copy it to %s" % (HDF5_FULL_PATH, HDF5_FOLDER))
if not os.path.exists(TAGS_FULL_PATH):
    sys.exit("The file %s doesn't exist. Please generate it using the serialize_dataset_to_hdf5.py file or copy it to %s. It should contain each label in one row separated by ','" % (TAGS_FULL_PATH, HDF5_FOLDER))
if not os.path.exists(MODEL_FOLDER_PATH):
    os.makedirs(MODEL_FOLDER_PATH)


###################
####   Class   ####
###################

# This is maybe not the best solution ever but I needed Keras ImageDataGenerator to take a batch generator to work fine with my configuration
# It may be necessary to update this override if we change our Keras version
# If Keras add a function for using generator in this generator, it will be a good idea to update to this version.
# @see https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
class CustomImageDataGenerator(ImageDataGenerator):
    def flow(self, custom_batch_generator, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png'):
        for x, y in custom_batch_generator:
            return NumpyArrayIterator(
                x, y, self,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                data_format=self.data_format,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format
            )

###################
#### Functions ####
###################

## Return an skf seeded instance
def new_kfold_seeded_instance():
    ## We use cross validation to increase the pertinence of our training and prevent it from generalizing
    # @see http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # The seed is important here because we need to call to generator for validation and training.
    # After each training session we just have to change the seed to rotate distribution.
    seed = random.randrange(4294967295)
    # The data are already shuffled in the dataset generation
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=False, random_state=seed)

## Custom data generator using cross validation and loading only segmented part of dataset if needed
def custom_batch_generator(skf, for_training=True):
    if DATA_SPLIT == 0 or DATA_SPLIT == DATA_COUNT:
        # Batch generator without data segmentation
        with h5py.File(HDF5_FULL_PATH, 'r') as f:
            data = f['my_data'][()]
            y = f['my_labels'][()]
            labels = np_utils.to_categorical(y, CLASSES_COUNT)

            if for_training: 
                for i, (train_index, test_index) in enumerate(skf.split(data, y)):
                    yield data[train_index], labels[train_index]
            else:
                for i, (train_index, test_index) in enumerate(skf.split(data, y)):
                    yield data[test_index], labels[test_index]
    else:
        # Batch generator with data segmentation
        with h5py.File(HDF5_FULL_PATH, 'r') as f:
            for i in range(DATA_SPLIT):
                if i == 0:
                    data = f['my_data'][:COUNT_SPLIT]
                    y = f['my_labels'][:COUNT_SPLIT]
                elif i == DATA_SPLIT-1:
                    data = f['my_data'][COUNT_SPLIT*i:]
                    y = f['my_labels'][COUNT_SPLIT*i:]
                else:
                    data = f['my_data'][COUNT_SPLIT*i:COUNT_SPLIT*(i+1)]
                    y = f['my_labels'][COUNT_SPLIT*i:COUNT_SPLIT*(i+1)]

                labels = np_utils.to_categorical(y, CLASSES_COUNT)

                if for_training: 
                    for i, (train_index, test_index) in enumerate(skf.split(data, y)):
                        yield data[train_index], labels[train_index]
                else:
                    for i, (train_index, test_index) in enumerate(skf.split(data, y)):
                        yield data[test_index], labels[test_index]
           

## Callback for logging model trends by epoch
class modelTrends(Callback):
    def on_epoch_end(self, epoch, logs={}):
        log = "Epoch %s : Loss = %f, Val_loss = %f | Acc = %f, Val_acc = %f | Top-5-acc : %f, Val_top-5-acc : %f" % (epoch, logs['loss'], logs['val_loss'], logs['categorical_accuracy'], logs['val_categorical_accuracy'], logs['top_k_categorical_accuracy'], logs['val_top_k_categorical_accuracy'])
        logger.log(log, 3)

## Training function used by our cross_validation process.
def train_model(model, filepath):
    # Early stopping if the validation loss (val_loss) doesn't decrease anymore
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_SOPPING_PATIENCE, verbose=1, mode='min')
    # We always keep the best model in case of early stopping
    model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)
    # We record model trends for each epochs using our custom callback
    model_trends = modelTrends()
    # We use a hack (just to let it take generator instead of X,Y) Keras data generator to do data-augmentation
    datagen = CustomImageDataGenerator(
        ImageDataGenerator(
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
    )
    # Cross Validation instance (seeded because we want to get the same things 2 time for validations/test)
    skf = new_kfold_seeded_instance()

    test_sample_count = int((1/N_SPLITS) * DATA_COUNT)
    validation_sample_count = DATA_COUNT - test_sample_count

    steps_per_epoch = validation_sample_count // BATCH_SIZE if validation_sample_count > BATCH_SIZE else validation_sample_count
    validation_steps = test_sample_count // BATCH_SIZE if test_sample_count > BATCH_SIZE else test_sample_count

    # Datagen contain a data augmentation generator defined below on the script
    # We hacked Keras ImageDataGenerator.flow to take batch_generator for our need. We describe why upper in the class method definition
    model.fit_generator(
        generator=datagen.flow(custom_batch_generator=custom_batch_generator(skf, for_training=True), batch_size=BATCH_SIZE, shuffle=True),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=datagen.flow(custom_batch_generator=custom_batch_generator(skf, for_training=False), batch_size=BATCH_SIZE),
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, model_trends]
    )

    # We reload the best epoch weight before keep going
    model.load_weights(filepath)
    # We clear stuff for memory safety
    early_stopping=model_checkpoint=model_trends=datagen=skf = None


################
#### Script ####
################

logger.header()

## We compute the needed value we need to load in ram only parts of the dataset at the same time
with open(TAGS_FULL_PATH, 'r') as f:
    tags = f.readline().strip().split(',')
CLASSES_COUNT = len(tags)

with h5py.File(HDF5_FULL_PATH, 'r') as f:
    DATA_COUNT = len(f['my_labels'][()])

if MAX_DATA_LOAD >= DATA_COUNT:
    DATA_SPLIT = 0 
else:
    DATA_SPLIT = math.ceil(DATA_COUNT / MAX_DATA_LOAD)
    COUNT_SPLIT = DATA_COUNT // DATA_SPLIT

#1# Original InceptionV3 model loading
model = net.build_model(CLASSES_COUNT)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
logger.log("The Original InceptionV3 model has been loaded", 0)


#2# Train the model on the new data for a few epochs and save
logger.log("Model first train, evaluation and save", 0)
first_train_start = time.time()

filepath = MODEL_FILE_FULL_PATH + "_0.h5"
train_model(model, filepath)

net.save(model, tags, MODEL_FILE_FULL_PATH + "_0")
logger.execution_time(first_train_start ,"Model first train, evaluation and save", 0)

#3# Fine-tuning convolutional layers from Inception V3
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

    train_model(model, filepath)

    # We save the best model for each Mega-Epoch        
    net.save(model, tags, MODEL_FILE_FULL_PATH + sufix)
    logger.execution_time(big_epoch_start, "Mega-epoch " + str(i), 2)

logger.execution_time(second_train_start, "Second training", 1)
logger.execution_time(finetuning_start ,"Fine-tuning convolutional layers from Inception V3", 0)
logger.footer(STARTED_DATETIME)
