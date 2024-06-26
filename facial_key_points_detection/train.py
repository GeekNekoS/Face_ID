import os

import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

global image_h
global image_w
global num_landmarks
global classes
global rgb_codes


def load_dataset(path):
    list1 = []
    for paths in os.walk(path + '/train' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/train/images/' + path1)
    train_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/train' + '/landmarks' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/train/landmarks/' + path1)
    train_y = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/val' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/val/images/' + path1)
    valid_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/val' + '/landmarks' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/val/landmarks/' + path1)
    valid_y = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/test' + '/images' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/test/images/' + path1)
    test_x = sorted(list1)
    list1.clear()
    for paths in os.walk(path + '/test' + '/landmarks' + '/'):
        for path1 in paths[2]:
            list1.append(path + '/test/landmarks/' + path1)
    test_y = sorted(list1)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image_mask(img_path, dots_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_shape = img.shape
    resized_img = cv2.resize(img, (image_w, image_h))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    resized_img = resized_img.astype(np.float32)
    resized_img = np.expand_dims(resized_img, -1)
    dots = []
    with open(dots_path, 'r') as dots_file:
        dots_file.readline()
        lines = dots_file.readlines()
        for line in lines:
            coordinates = line.strip().split()
            coordinates = (float(coordinates[0]) / img_shape[0], float(coordinates[1]) / img_shape[1])
            dots.append(coordinates[0])
            dots.append(coordinates[1])
    dots_array = np.array(dots, dtype=np.float32)
    return resized_img, dots_array


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)
    image, landmarks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    image.set_shape([image_h, image_w, 1])
    landmarks.set_shape(num_landmarks)
    return image, landmarks


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
    num_landmarks = 212
    input_shape = (image_h, image_w, 1)
    batch_size = 8
    lr = 1e-4 ## 0.0001
    num_epochs = 100
    dataset_path = "/kaggle/input/lapaaaa/LaPa"
    model_path = '/kaggle/working/FacialKeyPoints.keras'
    csv_path = '/kaggle/working/data.csv'
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")
    model = make_model(input_base(input_shape), num_landmarks)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.mean_absolute_error)
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