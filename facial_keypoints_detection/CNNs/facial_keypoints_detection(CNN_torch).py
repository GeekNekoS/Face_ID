import torch
import pandas as pd
import numpy as np
from facial_keypoints_detection.data_preprocessing.images_preprocessing import (preprocessing_X, preprocessing_y)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class FacialKeypointsDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(in_features=11*11*128, out_features=500)
        self.act4 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=500, out_features=500)
        self.act5 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(in_features=500, out_features=30)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_path = '../training_data/training.csv'
train = pd.read_csv(train_path)
X_train = torch.from_numpy(preprocessing_X(train, 96)).to(device)
train.drop('Image', axis=1, inplace=True)
y_train = torch.from_numpy(preprocessing_y(train)).to(device)
facial_keypoints_detect = FacialKeypointsDetection()
facial_keypoints_detect = facial_keypoints_detect.to(device)
optimizer = torch.optim.Adam(facial_keypoints_detect.parameters(), lr=0.001)
criterion = torch.nn.L1Loss()
batch_size = 50
epochs_n = 1
test_accuracy_history = []
test_loss_history = []
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


def learning(X_train, y_train, X_test, y_test):
    for epoch in range(epochs_n):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            batch_indexes = order[start_index:start_index+batch_size]
            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)
            preds = facial_keypoints_detect(X_batch)
            loss_value = criterion(preds, y_batch)
            loss_value.backward()
            optimizer.step()
        test_preds = facial_keypoints_detect.forward(X_test)
        test_loss_history.append(criterion(test_preds, y_test).data.cpu())
        accuracy = (abs(test_preds - y_test) < 1).float().mean().data.cpu()
        test_accuracy_history.append(accuracy)
        print(accuracy)


learning(X_train, y_train, X_test, y_test)
torch.save(facial_keypoints_detect, '../../facial_keypoints_detection(CNN_torch).pt')
plt.plot(test_loss_history)
plt.show()