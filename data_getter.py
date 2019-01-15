from __future__ import division, print_function

import base64, io, itertools, functools, json, os, random, re, textwrap, time
import numpy as np
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


def load_data_generator(x, y, batch_size=64):
    num_samples = x.shape[0]
    while 1:  # Loop forever so the generator never terminates
        try:
            shuffle(x)
            for i in range(0, num_samples, batch_size):
                x_data = [preprocess_image(im) for im in x[i:i + batch_size]]
                y_data = y[i:i + batch_size]

                # convert to numpy array since this what keras required
                yield shuffle(np.array(x_data), np.array(y_data))
        except Exception as err:
            print(err)

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

from sklearn.utils import shuffle


for i, name in enumerate(classes):
    print(name, end=' ')
    dst = '%s/%s.npy' % (data_path, name)
    image = np.load(dst)
    #image = image.reshape(image.shape[0], 28, 28)
    X.append(np.array(image))
    Y.append(keras.utils.to_categorical(np.full(image.shape[0], i), len(classes)))

#add shuffle on dataset 

Y_final = Y[0]
X_final = X[0]
for i, y in enumerate(Y):
    if i != 0:
        Y_final = np.concatenate((Y_final, y), axis=0)
        X_final = np.concatenate((X_final, X[i]), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.2, random_state=42)

input_tensor = Input(shape=(96,96,3))

base_model = MobileNetV2(input_tensor=input_tensor, input_shape=(96, 96, 3), include_top=False, weights='imagenet', classes=number_of_classes, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

op = Dense(256, activation='relu')(base_model.output)
output_tensor = Dense(number_of_classes, activation='softmax')(op)
model = Model(inputs=input_tensor, outputs=output_tensor)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
model.summary()


train_generator = load_data_generator(X_train, y_train, batch_size=64)
test_generator = load_data_generator(X_test, y_test, batch_size=64)



#training_history = model.fit_generator(
#    generator=train_generator,
#    steps_per_epoch=1000,
#    verbose=1,
#    epochs=100)
# model.save('zooModel.h5')

model = load_model('zooModel.h5')

# results = model.evaluate_generator(generator=test_generator, steps=len(X_test)/64, verbose=1)
results = model.evaluate_generator(generator=train_generator, steps=100, verbose=1)

print(results)