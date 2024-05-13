import keras


FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS = {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh}