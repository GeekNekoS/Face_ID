import keras
import pandas as pd
from SimpleCNN import SimpleCNN
import numpy as np
from CustomGenerator import CustomGenerator
from sklearn.model_selection import train_test_split
from facial_keypoints_detection.draw.draw_result import draw


pms = {
    'test_size': 0.2,
    'random_state': 0,
    'batch_size': 32,
    'epochs': 1,
}
stop_pms = {
    'monitor': 'val_loss',
    'patience': 10,
    'restore_best_weights': True,
}

model_1 = SimpleCNN((96, 96, 1))


def load_data(csv_path, columns=None, dropna=True):
    """""
    load data and normalize the images
    - csv_path: path of csv file containing the data
    - columns: columns to load. If none load all
    - dropna: if True drop data rows with missing labels

    return:
    - X: 2-d numpy array (n_samples, img_size**2)
    - y: 2-d numpy array (n_samples, n_landmarks*2) 
    """""

    test_set = csv_path.split('/')[-1] == 'test.csv'
    y = None

    df = pd.read_csv(csv_path)
    if columns != None:
        df = df[list(columns) + ['Image']]
    if dropna:
        df = df.dropna()
    if not test_set:
        y = df.drop(columns='Image').values

    X = df['Image'].apply(lambda x: np.fromstring(x, dtype='float', sep=' '))
    X = np.vstack(X) / 255.
    X = np.reshape(X, (len(X), 96, 96, 1))
    return X, y


dataset_path = '../data_for_learning/training.csv'
X, y = load_data(dataset_path)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=pms['test_size'], shuffle=True, random_state=pms['random_state'])


class Specialist:

    def __init__(self, source_model, spec_settings):
        """""
        arguments:
        - source_model: pretrained model
        - settings: list of dictionaries indicating model_name, columns to load from data
        """""
        super().__init__()
        self.source_model = source_model
        self.settings = spec_settings
        self.fitted_models = None

    def create_specialist(self, out_shape):
        """""
        create specialist model with transfer learning

        arguments:
        - out_shape: shape of specialist output layer

        return: 
        - model: single specialist model
        """""
        model = keras.Sequential()
        for layer in self.source_model.layers[:-1]:
            model.add(layer)
            layer.trainable = False
        model.add(keras.layers.Dense(out_shape, name='output'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, csv_path, pms, stop_pms):
        """""
        create specialist model with transfer learning

        arguments:
        - csv_path: training data path
        """""
        self.fitted_models = []
        for settings in self.settings:
            X, y = load_data(csv_path, columns=settings['cols'])
            generator = CustomGenerator(flip_indices=settings['x_flip'])
            X_new, y_new = generator.generate(X, y, shuffle=True)

            sub_model = self.create_specialist(out_shape=len(settings['cols']))
            history = sub_model.fit(X_new, y_new, validation_split=pms['test_size'], batch_size=pms['batch_size'], epochs=pms['epochs'], callbacks=keras.callbacks.EarlyStopping(**stop_pms))
            specialist = {'model_name': settings['model_name'], 'cols': settings['cols'], 'model': sub_model, 'history': history}
            self.fitted_models.append(specialist)

    def predict(self, X):
        """""
        make predictions

        arguments:
        - X: test images

        return: 
        - y_pred: predictions array (n_samples, 30)
        """""
        y_pred = pd.DataFrame()
        for specialist in self.fitted_models:
            sub_model = specialist['model']
            cols = list(specialist['cols'])
            y_pred[cols] = sub_model.predict(X)
        return y_pred.values


specialists_settings = [
    dict(model_name='eye_center',
         cols=('left_eye_center_x', 'left_eye_center_y',
               'right_eye_center_x', 'right_eye_center_y'),
         x_flip=[(0,2), (1,3)]),
    dict(model_name='eye_corner',
         cols=('left_eye_inner_corner_x', 'left_eye_inner_corner_y',
               'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
               'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
               'right_eye_outer_corner_x', 'right_eye_outer_corner_y'),
         x_flip=[(0,4), (1,5), (2,6), (3,7)]),
    dict(model_name='eyebrow_end',
         cols=('left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
               'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
               'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
               'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y'),
         x_flip=[(0,4), (1,5), (2,6), (3,7)]),
    dict(model_name='nose_tip',
         cols=('nose_tip_x', 'nose_tip_y'),
         x_flip=[(0,0), (1,1)]),
    dict(model_name='mouth',
         cols=('mouth_left_corner_x', 'mouth_left_corner_y',
               'mouth_right_corner_x', 'mouth_right_corner_y',
               'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
               'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'),
         x_flip=[(0,2), (1,3), (4,4), (5,5), (6,6), (7,7)]),
]
model_2 = Specialist(source_model=model_1, spec_settings=specialists_settings)
draw(X_val, model_2)
model_2.fit(csv_path=dataset_path, pms=pms, stop_pms=stop_pms)


