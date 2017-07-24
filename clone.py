import csv
import cv2
import numpy as np
import os
import sklearn

lines = []
driving_log_file = '/Users/test/Documents/Project3/CarND-Behavioral-Cloning-P3/training_data/driving_log.csv'

with open(driving_log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
from random import shuffle

# Reference: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
# Modify the brightness of an image
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255] = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# Generator to help load images in part to ease pressure of memory
def generator(samples, batch_size=64):
    num_samples = len(samples)
    correction = [0.0, 0.21, -0.21]

    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    image = cv2.imread(source_path)
                    measurement = float(batch_sample[3]) + correction[i]  # Steering angle

                    # Convert to RGB
                    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(imgRGB)
                    measurements.append(measurement)

                    # Flip RGB Image
                    images.append(cv2.flip(imgRGB, 1))
                    measurements.append(-1.0 * measurement)

                    # Modify Brightness of the image
                    imgBright = augment_brightness_camera_images(imgRGB)
                    images.append(imgBright)
                    measurements.append(measurement)

                    # Add Flipped Brightness image
                    images.append(cv2.flip(imgBright, 1))
                    measurements.append(-1.0 * measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# nVIDIA Model with
# normalization, 5 Convolution filters, followed by 4 Fully connected layers
model = Sequential()
model.add(Lambda(lambda x: ((x/255.0) - 0.5), input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use Mean-Squared error for loss, and adam optimizer
model.compile(loss='mse', optimizer='adam')

samples_per_epoch = 48000
history_object = model.fit_generator(train_generator,
                    samples_per_epoch= samples_per_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3,
                    verbose=2)

model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
