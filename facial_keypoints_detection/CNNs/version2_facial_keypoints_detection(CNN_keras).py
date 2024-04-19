import keras
import pandas as pd
from facial_keypoints_detection.data_preprocessing.images_preprocessing import (preprocessing_X, preprocessing_y)
from keras.optimizers import Adam


train_path = '../data_for_learning/training.csv'
train = pd.read_csv(train_path)
X_train = preprocessing_X(train, 96)
train.drop('Image', axis=1, inplace=True)
y_train = preprocessing_y(train)


model = keras.Sequential([
    keras.layers.Input((1, 96, 96), batch_size=32, dtype='float32'),
    keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', data_format='channels_first'),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Conv2D(64, (2, 2), strides=1, activation='relu', data_format='channels_first'),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Conv2D(128, (2, 2), strides=1, activation='relu', data_format='channels_first'),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(30),
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
history = model.fit(X_train.reshape(-1, 1, 96, 96), y_train, epochs=3, batch_size=32, validation_split=0.2)
model.save('facial_keypoints_detection(CNN, 3 epochs).keras')
