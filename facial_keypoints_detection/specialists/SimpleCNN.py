import keras


def model_constructor(input_shape):
    inputs = keras.layers.Input(input_shape)
    x = inputs
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    outputs = keras.layers.Dense(30)(x)
    model = keras.Model(inputs, outputs, name='SimpleCNN')
    model.compile(loss='mse', optimizer='adam')
    return model