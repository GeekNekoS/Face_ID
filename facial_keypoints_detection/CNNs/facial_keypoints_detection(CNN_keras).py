import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_X
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing_y


train_path = '../training_data/training.csv'
train = pd.read_csv(train_path)
X_train = preprocessing_X(train, 96)
train.drop('Image', axis=1, inplace=True)
y_train = preprocessing_y(train)

model = Sequential([
    Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', input_shape=(1, 96, 96), data_format='channels_first'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', data_format='channels_first'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', data_format='channels_first'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', data_format='channels_first'),
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
history = model.fit(X_train.reshape(-1, 1, 96, 96), y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save('facial_keypoints_detection(CNN, 3 epochs).keras')