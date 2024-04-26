import keras
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing
import pandas as pd
from facial_keypoints_detection.draw.draw_result import draw


test = '../data_for_learning/training.csv'
test_data = pd.read_csv(test)
model = keras.models.load_model('modelCNN(mae).h5', {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh})
X_train, y_train = preprocessing(test_data, 96)
results = model.predict(X_train)
for i in range(len(results)):
    print(y_train[i], results[i])
    draw(X_train[i], results[i])