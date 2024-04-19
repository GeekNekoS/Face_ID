import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_X
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_y


def model_constructor(inputA):
    dr = 0.3
    x = inputA
    x = Conv2D(32, kernel_size=3, activation='relu', data_format='channels_first')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(dr)(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(dr)(x)
    x = Dense(30, activation='relu')(x)
    return x


def pipeline(path: str, batch_size: int) -> tf.data.Dataset:
    dataset = pd.read_csv(path)
    X = preprocessing_X(dataset, 96)
    dataset.drop('Image', axis=1, inplace=True)
    y = preprocessing_y(dataset)
    zipped_input = tf.data.Dataset.zip((X, y)).batch(batch_size)
    return zipped_input


train_path = '../data_for_learning/training.csv'
test_path = '../test/test.csv'
batch_size = 32
input_shape = (1, 96, 96)

train = pipeline(train_path, batch_size)
test = pipeline(test_path, batch_size)

inputA = Input(shape=input_shape, name='input_image')
final = model_constructor(inputA)
model = Model(inputs=[inputA], outputs=final)

optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
# history = model.fit(X_train.reshape(-1, 1, 96, 96), y_train, epochs=10, batch_size=32, validation_split=0.2)
history = model.fit(train, validation_data=test, epochs=10, validation_split=0.2, verbose=1)

model.save('facial_keypoints_detection(CNN, 3 epochs).keras')
