from __future__ import division, print_function

import base64, io, itertools, functools, json, os, random, re, textwrap, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

X = []
Y = []

target_size = 96

def preprocess_image(x):
    x = resize(x, (target_size, target_size),
            mode='constant',
            anti_aliasing=False)
    x = np.stack((x,)*3, axis=-1) 
    return x.astype(np.float32)

#sess = tf.InteractiveSession()

for i, name in enumerate(classes):
    print(name, end=' ')
    dst = '%s/%s.npy' % (data_path, name)
    image = np.load(dst)
    #image = image.reshape(image.shape[0], 28, 28)
    images = []
    for j in range(10000):
        x = image[j]
        x = resize(x, (target_size, target_size),mode='constant',anti_aliasing=False)
        x = np.stack((x,)*3, axis=-1) 
        x = x.astype(np.float32)
        images.append(x)
    X.append(np.array(images))
    Y.append(keras.utils.to_categorical(np.full(10000, i), len(classes)))

#add shuffle on dataset 

Y_final = Y[0]
X_final = X[0]
for i,y in enumerate(Y):
    if i != 0:
        Y_final = np.concatenate((Y_final, y), axis=0)
        X_final = np.concatenate((X_final, X[i]), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.2, random_state=42)

input_tensor = Input(shape=(96,96,3))

base_model = MobileNetV2()

base_model = MobileNetV2(input_tensor=input_tensor, input_shape=(96, 96, 3), include_top=False, weights='imagenet', classes=number_of_classes, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
#x = Flatten()(x)
#x = Dense(512, activation='relu')(x) # adding just this worked best so far proly depends on number of classes as well
predictions = Dense(number_of_classes, activation='softmax',use_bias=True, name='Logits')(x)
'''
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape_2')(x)
'''


#predictions = Dense(number_of_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=100)