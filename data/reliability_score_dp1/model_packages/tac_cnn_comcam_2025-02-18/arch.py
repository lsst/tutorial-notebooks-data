__all__ = ["TACCNN"]

import torch
import torch.nn as nn


class TACCNN(nn.Module):
    def __init__(self, input_shape=(3, 51, 51)):
        super(TACCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)

        dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
        self.forward_conv(dummy_input)
        self.fc1 = nn.Linear(self.num_shape[1], 32)
        self.fc2 = nn.Linear(32, 1)

    def forward_conv(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # x = self.dropout2(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        self.num_shape = x.size()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.sigmoid(x).squeeze(1)
