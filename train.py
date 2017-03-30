#%%
'''Reads traffic sign data for German Traffic Sign Recognition Benchmark. '''
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

TRAIN_PATH = r".\Final_Training\Images"
images = [] # images
labels = [] # corresponding labels
# loop over all 43 classes
for c in range(0, 43):
    prefix = join(TRAIN_PATH, str(c).zfill(5)) # subdirectory for class
    gtFile = join(prefix, 'GT-' + str(c).zfill(5) + '.csv') # annotations file
    dataFrame = pd.read_csv(gtFile, delimiter=';')
    # loop over all images in current annotations file
    for _, row in dataFrame.iterrows():
        img = plt.imread(join(prefix, row['Filename'])) # the 1th column is the filename
        img = img[row['Roi.X1']:row['Roi.X2'], row['Roi.Y1']:row['Roi.Y2'], :]
        img = imresize(img, (32, 32))
        images.append(img)
        labels.append(row.ClassId) # the 8th column is the label

X = np.stack(images)
Y = np.asarray(labels)


#%%
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

nb_classes = 43
nb_filters = 64
kernel_size = (3, 3)
pool_size = (2, 2)
input_shape = (32, 32, 3)

X = X.astype('float32')
X /= 255
Y = np_utils.to_categorical(Y, nb_classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model = Model(input=model.inputs, output=model.outputs)
model.load_weights(r'.\cnn.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, nb_epoch=16,
          validation_data=(X_test, Y_test), initial_epoch=10)

model.save(r'.\cnn.h5')
