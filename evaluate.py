"""Eval cnn Model"""
#%%
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imresize

TEST_PATH = r".\Final_Test\Images"

gtFile = join(TEST_PATH, 'GT-final_test' + '.csv') # annotations file
dataFrame = pd.read_csv(gtFile, delimiter=';')
images = []
labels = []
for _, row in dataFrame.iterrows():
    img = plt.imread(join(TEST_PATH, row['Filename'])) # the 1th column is the filename
    img = img[row['Roi.X1']:row['Roi.X2'], row['Roi.Y1']:row['Roi.Y2'], :]
    img = imresize(img, (32, 32))
    images.append(img)
    labels.append(row.ClassId) # the 8th column is the label

nb_classes = 43
X = np.stack(images)
X = X.astype('float32')
X /= 255
Y = np.asarray(labels)
#from keras.utils import np_utils
#Y = np_utils.to_categorical(Y, nb_classes)

from keras.models import load_model
model = load_model(r".\cnn.h5")

res = model.predict(X, batch_size=128, verbose=1)
pred_Y = np.argmax(res, axis=1)
print("Accuracy: {}".format(sum(Y==pred_Y)/Y.size))
