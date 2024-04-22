from keras import (layers, Model)


def SimpleCNN(input_shape):
    n_layers = 3
    n_filters = [32 * 2**i for i in range(n_layers)]
    inputs = layers.Input(input_shape, name='input')
    x = inputs
    for j, filters in enumerate(n_filters):
        x = layers.Conv2D(filters, kernel_size=(3,3), activation='relu', name=f'conv_{j+1}')(x)
        x = layers.MaxPooling2D(name=f'pooling_{j+1}')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.Dense(512, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(30, name='output')(x)
    model = Model(inputs, outputs, name='SimpleCNN')
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model