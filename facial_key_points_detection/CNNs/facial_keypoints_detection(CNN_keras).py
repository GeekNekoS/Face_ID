import pandas as pd
import keras
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from facial_key_points_detection.data_preprocessing.images_preprocessing import make_pipeline


def model_constructor(inputA):
    dr = 0.1
    x = inputA
    x = Conv2D(32, 3, 1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, 1, activation='relu', padding='same')(x)
    x = Dropout(dr)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, 3, 1, activation='relu', padding='same')(x)
    x = Dropout(dr)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(256, 3, 1, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(30)(x)
    return x


dataset_path = '../data_for_learning/training.csv'
dataset = pd.read_csv(dataset_path)
batch_size = 32
input_shape = (96, 96, 1)

train, val = make_pipeline(dataset, 96, 96, batch_size)

inputA = Input(shape=input_shape, name='input_image')
final = model_constructor(inputA)
model = Model(inputs=[inputA], outputs=final)
print(model.summary())

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=keras.losses.mean_absolute_error, metrics=[keras.losses.huber, keras.losses.log_cosh])
my_callbacks = [keras.callbacks.ModelCheckpoint(filepath='modelCNN(mae).h5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)]
history = model.fit(train, validation_data=val, epochs=15, verbose=1, callbacks=my_callbacks)
