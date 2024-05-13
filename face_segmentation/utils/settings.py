import keras


U_NET_CUSTOM_OBJECTS = {'categorical_crossentropy': keras.losses.categorical_crossentropy}