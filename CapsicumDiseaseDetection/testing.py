import os
import cv2
# from skimage import io
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

data = np.load('Processed_Data/data.npy')
target = np.load('Processed_Data/target.npy')

# In[2]:


from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

# In[3]:



model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('model.h5')



test_path = 'test'

img_names = os.listdir(test_path)

category = {'Phytophthora Blight': 0, 'Powdery Mildew': 1, 'Cercospora Leaf Spot': 2}

img = cv2.imread('test/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (50, 50))
normalized = resized / 255.0
reshaped = np.reshape(normalized, (1, 50, 50, 1))

result = model.predict(reshaped)
label = np.argmax(result, axis=1)[0]
prob = np.max(result, axis=1)[0]
prob = round(prob, 2) * 100

print(result)
print(np.argmax(result, axis=1))
print(np.max(result, axis=1)[0])


