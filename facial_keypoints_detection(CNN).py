import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam


train_path = 'facial-keypoints-detection/training/training.csv'
test_path = 'facial_keypoints_detection/test/test.csv'
id_path = 'facial-keypoints-detection/IdLookupTable.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
id = pd.read_csv(id_path)
train.ffill(inplace=True)
m, n = train.shape
img = []
img_size = 96
for i in range(m):
    splitting = np.array(train['Image'][i].split(' '), dtype='float64')
    splitting = np.reshape(splitting, (img_size, img_size, 1))
    splitting /= 255
    img.append(splitting)
img = np.array(img)
X_train = img
train.drop('Image', axis=1, inplace=True)
y_train = []
for i in range(len(train)):
    y = train.iloc[i, :].values
    y_train.append(y)
y_train = np.array(y_train, dtype='float')
model = Sequential([
    Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', input_shape=(96, 96, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), strides=1, activation='relu', padding='same'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), strides=1, activation='relu', padding='same'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), strides=1, activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(30)
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
history = model.fit(X_train.reshape(-1, 96, 96, 1), y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save('facial_keypoints_detection(CNN, 3 epochs).h5')