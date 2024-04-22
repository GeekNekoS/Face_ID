import keras
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing
import pandas as pd


test = '../data_for_learning/training.csv'
test_data = pd.read_csv(test)
model = keras.models.load_model('model.01-263.55.keras')
X_train, y_train = preprocessing(test_data, 96)
model.predict(X_train)