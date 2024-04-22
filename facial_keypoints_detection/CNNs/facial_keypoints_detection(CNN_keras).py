import pandas as pd
import keras
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from facial_keypoints_detection.data_preprocessing.images_preprocessing import make_pipeline


def model_constructor(inputA):
    dr = 0.1
    x = inputA
    x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(30, activation='relu')(x)
    return x


dataset_path = '../data_for_learning/training.csv'
dataset = pd.read_csv(dataset_path)
batch_size = 32
input_shape = (96, 96, 1)

train, val = make_pipeline(dataset, 96, batch_size, augmentation=True)

inputA = Input(shape=input_shape, name='input_image')
final = model_constructor(inputA)
model = Model(inputs=[inputA], outputs=final)
print(model.summary())

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
my_callbacks = [keras.callbacks.ModelCheckpoint(filepath='modelCNN.keras', monitor='mse', verbose=1, save_best_only=True, mode='min')]
history = model.fit(train, validation_data=val, epochs=1, verbose=1, callbacks=my_callbacks)
