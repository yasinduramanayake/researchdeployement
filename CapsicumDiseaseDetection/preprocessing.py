import os
import cv2
import numpy as np

path = 'Data'

categories = os.listdir(path)
labels = list(range(len(categories)))

categorDict = dict(zip(categories, labels))

print(categories)
print(labels)
print(categorDict)

size = 50
dataset = []

for category in categories:
    imgs_path = os.path.join(path, category)
    imgs_names = os.listdir(imgs_path)

    for img_name in imgs_names:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)

        try:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (size, size))
            dataset.append([gray, categorDict[category]])

        except Exception as e:

            print(e)

len(dataset)

from random import shuffle

shuffle(dataset)

data = []
target = []

for img, label in dataset:
    data.append(img)
    target.append(label)

data = np.array(data)
target = np.array(target)

data = data.reshape(data.shape[0], size, size, 1) / 255

from keras.utils import np_utils

target = np_utils.to_categorical(target)

print(data.shape)
print(target.shape)

np.save('Processed_Data/data', data)
np.save('Processed_Data/target', target)
