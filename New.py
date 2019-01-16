from __future__ import division, print_function

import base64, io, itertools, functools, json, os, random, re, textwrap, time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image, ImageDraw
from six.moves.urllib import request
from xml.dom import minidom
from tensorflow import keras
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import MobileNetV2
import random
np.random.seed(1337)


def list_bucket(bucket, regexp='.*'):
    """Returns a (filtered) list of Keys in specified GCE bucket."""
    keys = []
    fh = request.urlopen('https://storage.googleapis.com/%s' % bucket)
    content = minidom.parseString(fh.read())
    for e in content.getElementsByTagName('Contents'):
        key = e.getElementsByTagName('Key')[0].firstChild.data
        if re.match(regexp, key):
            keys.append(key)
    return keys

#all_ndjsons = list_bucket('quickdraw_dataset', '.*npy$$')
#print('available: (%d)' % len(all_ndjsons))
#print('\n'.join(textwrap.wrap(
#    ' '.join([key.split('/')[-1].split('.')[0] for key in all_ndjsons]),
#    width=100)))

# Store all data locally in this directory.
data_path = '../data_cats_dogs'

# Mini group of two animals.
pets = ['dog','cat']

# Somewhat larger group of zoo animals.
zoo = ['elephant', 'giraffe', 'kangaroo', 'lion', 'monkey', 'panda',
       'penguin', 'rhinoceros', 'tiger', 'zebra']

animals = ['bat', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab',
           'crocodile', 'dog', 'dolphin', 'duck', 'elephant', 'fish',
           'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
           'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl',
           'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon',
           'rhinoceros', 'scorpion', 'sea turtle', 'shark', 'sheep',
           'snail', 'spider', 'squirrel', 'teddy-bear', 'tiger',
           'whale', 'zebra']

classes, classes_name = zoo, 'pets'

def retrieve(bucket, key, filename):
    """Returns a file specified by its Key from a GCE bucket."""
    url = 'https://storage.googleapis.com/%s/%s' % (bucket, key)
    if not os.path.isfile(filename):
        request.urlretrieve(url=url, filename=filename)

if not os.path.exists(data_path):
    os.mkdir(data_path)

number_of_classes = len(classes)

print('\n%d classes:' % number_of_classes)

for name in classes:
    print(name, end=' ')
    dst = '%s/%s.npy' % (data_path, name)
    retrieve('quickdraw_dataset', 'full/numpy_bitmap/%s.npy' % name, dst)
    print('%.2f MB' % (os.path.getsize(dst) / 2.**20))

print('\nDONE :)')

#!ls -lh $data_path

X = []
Y = []

target_size = 96

def preprocess_image(images):
    resizeImages = []
    return resizeImages


classSamples = 10000

X_final = []
Y_final = []

for i, name in enumerate(classes):
    print(name, end=' ')
    dst = '%s/%s.npy' % (data_path, name)
    images = np.load(dst)
    idx = np.random.choice(np.arange(len(images)), classSamples, replace=False)

    print("new class of images")
    for image_i in images[idx]:
        original = image_i.reshape(28, 28).astype('float32') / 255
        originalUpsized = resize(original, (target_size, target_size), anti_aliasing=False)
        tensor = np.stack((originalUpsized,) * 3, axis=-1)
        '''''
        plt.subplot(3, 1, 1)
        plt.imshow(tensor[:, :, 0])
        plt.subplot(3, 1, 2)
        plt.imshow(tensor[:, :, 1])
        plt.subplot(3, 1, 3)
        plt.imshow(tensor[:, :, 2])
        plt.show()
        '''''
        X_final.append(tensor)
        Y_final.append(keras.utils.to_categorical(i, num_classes=len(classes)))

X_final = np.array(X_final)
Y_final = np.array(Y_final)

x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.2, random_state=42)


input_tensor = Input(shape=(96,96,3))

base_model = MobileNetV2(input_tensor=input_tensor, input_shape=(96, 96, 3), include_top=False, weights='imagenet', classes=number_of_classes, pooling='avg')

####
trainableLayers = False
####

for layer in base_model.layers:
    layer.trainable = trainableLayers

op = Dense(256, activation='relu')(base_model.output)
output_tensor = Dense(number_of_classes, activation='softmax')(op)
model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
model.summary()

batch_size = 50

training_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10, verbose=1, callbacks=None, validation_split=0.25, shuffle=True)

f, ax = plt.subplots(1)
ax.plot(training_history.epoch, training_history.history["categorical_accuracy"], label="Train")
ax.plot(training_history.epoch, training_history.history["val_categorical_accuracy"], label="Val")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(str(trainableLayers) + ".jpg")

results = model.evaluate(x=x_test, y=y_test, verbose=1)
print("Test results: " + str(results[1]))