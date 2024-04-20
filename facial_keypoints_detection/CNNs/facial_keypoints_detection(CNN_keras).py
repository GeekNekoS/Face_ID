import pandas as pd
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from facial_keypoints_detection.data_preprocessing.images_preprocessing import make_pipeline


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
    x = Dropout(dr)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dr)(x)
    x = Dense(30, activation='relu')(x)
    return x


dataset_path = '../data_for_learning/training.csv'
dataset = pd.read_csv(dataset_path)
batch_size = 32
input_shape = (1, 96, 96)

train, val = make_pipeline(dataset, 96, batch_size, augmentation=True)

inputA = Input(shape=input_shape, name='input_image')
final = model_constructor(inputA)
model = Model(inputs=[inputA], outputs=final)

optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
# history = model.fit(X_train.reshape(-1, 1, 96, 96), y_train, epochs=10, batch_size=32, validation_split=0.2)
history = model.fit(train, validation_data=val, epochs=10, verbose=1)

model.save('facial_keypoints_detection(CNN, 3 epochs).keras')
