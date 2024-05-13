import keras


def load_model(path, custom_objects):
    model = keras.models.load_model(path, custom_objects)
    return model
