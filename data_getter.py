from __future__ import division, print_function

import base64, io, itertools, functools, json, os, random, re, textwrap, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from six.moves.urllib import request
from xml.dom import minidom
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import MobileNetV2


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

all_ndjsons = list_bucket('quickdraw_dataset', '.*npy$$')
print('available: (%d)' % len(all_ndjsons))
print('\n'.join(textwrap.wrap(
    ' '.join([key.split('/')[-1].split('.')[0] for key in all_ndjsons]),
    width=100)))

# Store all data locally in this directory.
data_path = '../data_cats_dogs'

# Mini group of two animals.
pets = ['dog']

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

classes, classes_name = pets, 'pets'

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

train_classes = []

sess = tf.InteractiveSession()

for name in classes:
    print(name, end=' ')
    dst = '%s/%s.npy' % (data_path, name)
    image = np.load(dst)
    print(image.shape)
    image_reshaped = image.reshape(image.shape[0], 28, 28, 1)
    print(image_reshaped.shape)
    #sess = tf.InteractiveSession()
    image_padded = tf.pad(image_reshaped, [[0, 0], [2,2], [2,2], [0, 0]]).eval()
    print(image_padded.shape)
    train_classes.append(image_padded)

for c in train_classes:
    print(c.shape)

base_model = MobileNetV2(input_shape=(32, 32, 1), include_top=False, weights=None, classes=number_of_classes)
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_of_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)