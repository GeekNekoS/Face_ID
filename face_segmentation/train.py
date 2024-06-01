import os

import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

global image_h
global image_w
global num_classes
global classes
global rgb_codes


def load_dataset(path):
    list1 = []
    for paths in os.walk(path + '/train' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/train/images/' + path1)
    train_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/train' + '/labels' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/train/labels/' + path1)
    train_y = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/val' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/val/images/' + path1)
    valid_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/val' + '/labels' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/val/labels/' + path1)
    valid_y = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/test' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/test/images/' + path1)
    test_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/test' + '/labels' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/test/labels/' + path1)
    test_y = sorted(list1)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image_mask(x, y):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h))
    y = y.astype(np.int32)
    return x, y


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)
    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])
    return image, mask


def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess)
    ds = ds.batch(batch).prefetch(2)
    return ds


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    image_h = 512
    image_w = 512
    num_classes = 11
    input_shape = (image_h, image_w, 3)
    batch_size = 8
    lr = 1e-4 ## 0.0001
    num_epochs = 100
    dataset_path = "/kaggle/input/lapaaaa/LaPa"
    model_path = '/kaggle/working/Unet.keras'
    csv_path = '/kaggle/working/data.csv'
    rgb_codes = [
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
    ]
    classes = [
        "background", "skin", "left eyebrow", "right eyebrow",
        "left eye", "right eye", "nose", "upper lip", "inner mouth",
        "lower lip", "hair"
    ]
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")
    model = unet(input_shape, num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]
    model.fit(train_ds,
              validation_data=valid_ds,
              epochs=num_epochs,
              callbacks=callbacks,
              verbose=1
    )