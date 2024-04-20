import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_X
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_y
from keras import callbacks
from keras import applications


def model_constructor(inputA):
    dr = 0.1
    x = inputA
    x = Conv2D(32, kernel_size=3, activation='relu', data_format='channels_first', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding = 'same')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(dr)(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(dr)(x)
    x = Dense(30, activation='relu')(x)
    return x


def pipeline(path: str) -> tf.data.Dataset:
    dataset = pd.read_csv(path)
    X = preprocessing_X(dataset, 96)
    dataset.drop('Image', axis=1, inplace=True)
    y = preprocessing_y(dataset)
    return X, y


train_path = '../data_for_learning/training.csv'
batch_size = 32
input_shape = (1, 96, 96)

X_train, y_train = pipeline(train_path)

inputA = Input(shape=input_shape, name='input_image')
final = model_constructor(inputA)
model = Model(inputs=[inputA], outputs=final)

model.compile(optimizer=Adam(0.001), loss='mae', metrics=['accuracy'])

my_callbacks = [
    callbacks.ModelCheckpoint(
        filepath='best_accuracy_facial_keypoints_detection_{epoch:03d}-{val_accuracy:03f}.keras',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True),
]

history = model.fit(X_train.reshape(-1, 1, 96, 96), y_train, epochs=1000, batch_size=batch_size,
                    validation_split=0.2, callbacks=my_callbacks, verbose=1)

model.save('facial_keypoints_detection(CNN, 100 epochs).keras')
