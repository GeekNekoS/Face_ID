from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Input


def input_base(input_shape, name='input_image'):
    return Input(input_shape, name=name)


def model_constructor(inputA, num_landmarks):
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
    x = Dense(num_landmarks)(x)
    return x


def make_model(input_layers, num_landmarks):
    return Model(inputs=[input_layers], outputs=model_constructor(input_layers, num_landmarks))
