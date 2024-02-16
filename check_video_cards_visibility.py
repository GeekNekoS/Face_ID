import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Видеокарта(ы) найдена(ы)")
    print(gpus)
else:
    print("Не вижу видеокарт!")
