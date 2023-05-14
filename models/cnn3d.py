
import torch.nn as nn

class cnn3d(nn.Module):

    def __init__(self, out_dim, p = 0.3):
        """
        3D CNN implemented by 
            https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/cnn.py

        Takes (batchsize, num_channel, 101, 101, 101) volume input
        """

        super(cnn3d, self).__init__()
        self.conv1 = self._conv_layer_set(1, 16)
        self.conv2 = self._conv_layer_set(16, 32)
        self.conv3 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(10*10*10*64, 128) # need to change input features for image shape
        self.fc2 = nn.Linear(128, out_dim)
        self.relu = nn.LeakyReLU()
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=p)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=0,
                ),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x
