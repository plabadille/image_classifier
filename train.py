import sys
import datetime, time

import numpy as np
from collections import defaultdict

# Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
# Then we need to import it before Keras
import scipy.misc

from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

import dataset
import net
import logger

np.random.seed(8585)
supplied_args = sys.argv[1:]

#-Constants
## Filesystem const
STARTED_DATETIME = datetime.datetime.now()
DATA_DIRECTORY = supplied_args[0] if supplied_args else "data_dir"
MODEL_FILE_PREFIX = supplied_args[1] if len(supplied_args) >= 2 else "image-classifier_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
## Dataset const
N = 224 #height/width for the images : InceptionV3 model require 224
CHANNELS = 3
## Training const
BATCH_SIZE = 32
EPOCHS = 15
BIG_EPOCHS = 2
PERCENT_OF_DATA_USED_FOR_TRAINING = 0.8

#### Needed function ####

def evaluate(model, vis_filename=None):
    Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(Y_pred, axis=1)

    accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    print("accuracy:", accuracy)
    
    confusion = np.zeros((classes_count, classes_count), dtype=np.int32)
    for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
        confusion[predicted_index, actual_index] += 1
    
    print("rows are predicted classes, columns are actual classes")
    for predicted_index, predicted_tag in enumerate(tags):
        print(predicted_tag[:7])
        for actual_index, actual_tag in enumerate(tags):
            print("\t%d" % confusion[predicted_index, actual_index])
        print()
    if vis_filename is not None:
        bucket_size = 10
        image_size = n // 4 # right now that's 56
        vis_image_size = classes_count * image_size * bucket_size
        vis_image = 255 * np.ones((vis_image_size, vis_image_size, CHANNELS), dtype='uint8')
        example_counts = defaultdict(int)
        for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, y_test, X_test):
            example_count = example_counts[(predicted_tag, actual_tag)]
            if example_count >= bucket_size**2:
                continue
            image = image.transpose((1, 2, 0))
            image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
            tilepos_x = bucket_size * predicted_tag
            tilepos_y = bucket_size * actual_tag
            tilepos_x += example_count % bucket_size
            tilepos_y += example_count // bucket_size
            pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
            vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
            example_counts[(predicted_tag, actual_tag)] += 1
        vis_image[::image_size * bucket_size, :] = 0
        vis_image[:, ::image_size * bucket_size] = 0
        scipy.misc.imsave(vis_filename, vis_image)


#### Script ####

logger.header()

#-Dataset gathering and formating
logger.log("Dataset gathering and formating", 0)
dataset_start = time.time()

X, y, tags = dataset.dataset(DATA_DIRECTORY, N)

classes_count = len(tags)
sample_count = len(y)
train_size = int(sample_count * PERCENT_OF_DATA_USED_FOR_TRAINING)

X_train = X[:train_size]
y_train = y[:train_size]
Y_train = np_utils.to_categorical(y_train, classes_count)
X_test  = X[train_size:]
y_test  = y[train_size:]
Y_test = np_utils.to_categorical(y_test, classes_count)
X, y = None, None # Performances cleanup (the value store in X are hudge)
logger.execution_time(dataset_start ,"Dataset gathering and formating", 0)

#- If the dataset and number of classes are low, it's recommanded to use an ImageDataGenerator.
# @see https://keras.io/preprocessing/image/

# Else I unrecommanded it unless you have a lot of ram (32Go is not suffisant for 101 classes with 1k images per class without using a lot of swap and then reduce significantly the speed of the gpu training).

#from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     rotation_range=0,
#     width_shift_range=0.125,
#     height_shift_range=0.125,
#     horizontal_flip=True,
#     vertical_flip=False,
#     fill_mode='nearest',
# )
# datagen.fit(X_train)

# Then you can use it directly in the model.fit_generator (insted of model.fit) like this : generator=datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)

#- Original InceptionV3 model loading
logger.log("Original InceptionV3 model loading", 0)
inception_start = time.time()

model = net.build_model(classes_count)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

logger.execution_time(inception_start ,"Original InceptionV3 model loading", 0)
die()
#- Train the model on the new data for a few epochs then evaluate and save
logger.log("Model first train, evaluation and save", 0)
first_train_start = time.time()

model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, Y_test),
)

evaluate(model, "000.png")
net.save(model, tags, model_file_prefix)

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

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"]) # we use SGD with a low learning rate

logger.execution_time(compile_start, "Recompiling the model to take our modifications", 1)

## We do the training
logger.log("Second training", 1)
second_train_start = time.time()

for i in range(1,BIG_EPOCHS):
    print("mega-epoch %d/%d" % i,BIG_EPOCHS)
    logger.log("Mega-epoch " + str(i), 2)
    big_epoch_start = time.time()

    model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, Y_test),
    )

    evaluate(model, str(i).zfill(3)+".png")
    net.save(model, tags, model_file_prefix)

    logger.execution_time(big_epoch_start, "Mega-epoch " + str(i), 2)

logger.execution_time(second_train_start, "Second training", 1)

logger.execution_time(finetuning_start ,"Fine-tuning convolutional layers from Inception V3", 0)

logger.footer(STARTED_DATETIME)
