import numpy as np

data = np.load('Processed_Data/data.npy')
target = np.load('Processed_Data/target.npy')


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

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


from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)


history = model.fit(train_data, train_target, epochs=100, validation_split=0.1)


from matplotlib import pyplot as plt

plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_loss'], 'r')


from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'], 'b')
plt.plot(history.history['val_accuracy'], 'r')


print(model.evaluate(test_data, test_target))


model.save_weights('model.h5')
