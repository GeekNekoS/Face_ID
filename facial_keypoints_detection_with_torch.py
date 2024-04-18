import torch


class FacialKeypointsDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), padding=(2, 2))
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=(2, 2))
        self.act2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), padding=(2, 2))
        self.act3 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=500, out_features=500)
        self.act4 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=500, out_features=30)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x


Titanic = FacialKeypointsDetection()
optimizer = torch.optim.Adam(Titanic.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()


def learning(X_train, y_train):
    for epoch in range(10):
       for batch_idx, (data, target) in enumerate(X_train, y_train):
           data, target = torch.Variable(data), torch.Variable(target)
           optimizer.zero_grad()
           net_out = Titanic(data)
           loss = criterion(net_out, target)
           loss.backward()
           optimizer.step()
