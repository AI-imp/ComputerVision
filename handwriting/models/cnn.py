import torch.nn as nn
class Netn(nn.Module):
    def __init__(self):
        super(Netn, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(3*28*28,3*256),
            nn.ReLU(),
            nn.Linear(3*256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )
        self.fc1=nn.Linear(3*28*28,3*256)
        self.fc2=nn.Linear(3 * 256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.dense(x)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  #产生32通道，3*3的卷积核
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # 32x14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  #产生64通道，3*3的卷积核
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 64x7x7
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x3x3
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # fc4 64*3*3 -> 128
            nn.ReLU(),
            nn.Linear(128, 10)  # fc5 128->10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)  # 64x3x3
        res = conv3_out.view(conv3_out.size(0), -1)  # batch x (64*3*3)
        out = self.dense(res)
        return out
