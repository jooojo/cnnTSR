"""Run cnn prediction"""
#%%
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.misc import imresize

TEST_PATH = r".\Final_Test\Images"

gt_file = join(TEST_PATH, 'GT-final_test' + '.csv') # annotations file
data_frame = pd.read_csv(gt_file, delimiter=';')
row = data_frame.xs(int(80))
print(row)
img = plt.imread(join(TEST_PATH, row['Filename'])) # the 1th column is the filename
img = img[row['Roi.X1']:row['Roi.X2'], row['Roi.Y1']:row['Roi.Y2'], :]
img = imresize(img, (32, 32))

from keras.models import load_model
model = load_model(r".\cnn.h5")
x = np.expand_dims(img, axis=0)
x = x.astype('float32')
x /= 255
prediction = model.predict(x)
classid = prediction.argmax()
print(classid, prediction[0][classid])

plt.imshow(img)
plt.show()
