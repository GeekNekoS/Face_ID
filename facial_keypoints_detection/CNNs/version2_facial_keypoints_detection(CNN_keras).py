import keras
import pandas as pd
from facial_keypoints_detection.data_preprocessing.images_preprocessing import make_pipeline
from keras.optimizers import Adam
import tensorflow as tf


batch_size = 32
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
   tf.config.experimental.set_memory_growth(gpu_devices[0], True)


model = keras.Sequential([
    keras.layers.Input((1, 96, 96), batch_size=32, dtype='float32'),
    keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first'),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Conv2D(128, (3, 3), strides=1, activation='relu'),
    keras.layers.MaxPooling2D((2, 2), strides=1),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(30),
])


dataset_path = '../data_for_learning/training.csv'
dataset = pd.read_csv(dataset_path)
train, val = make_pipeline(dataset, 96, batch_size)

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
history = model.fit(train, epochs=100, batch_size=batch_size, validation_data=val)
model.save('facial_keypoints_detection(CNN, 3 epochs).keras')
