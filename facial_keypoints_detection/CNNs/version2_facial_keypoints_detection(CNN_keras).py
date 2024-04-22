import keras
import pandas as pd
from facial_keypoints_detection.data_preprocessing.images_preprocessing import make_pipeline
from keras.optimizers import Adam
import tensorflow as tf


batch_size = 32
input_shape = (96, 96, 1)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


model = keras.Sequential([
    keras.layers.Input(input_shape, batch_size=32, dtype='float32'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(30),
])
print(model.summary())

dataset_path = '../data_for_learning/training.csv'
dataset = pd.read_csv(dataset_path)
train, val = make_pipeline(dataset, 96, batch_size, augmentation=True)

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
# history = model.fit(train, epochs=100, batch_size=batch_size, validation_data=val)
# model.save('facial_keypoints_detection(CNN, 3 epochs).keras')


# my_callbacks = [
#     keras.callbacks.ModelCheckpoint(filepath=filepath,  save='max')
# ]
# model.fit(dataset, epochs=1, callbacks=my_callbacks)


EPOCHS = 1
filepath = 'model.{epoch:02d}-{mse:.2f}.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor='mse',
    mode='max',
    save_best_only=True)

model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=[model_checkpoint_callback])


# keras.models.load_model(filepath)
