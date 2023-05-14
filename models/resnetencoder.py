import torch.nn.functional as F
from torch.nn import nn

class ResnetEncoder(nn.Module):
    """
    ResNet18 feature extractor
    """
    def __init__(self, num_channels, num_classes, weights = None, out_dim=20):
        super(ResnetEncoder, self).__init__()
        self.resnet = resnet18(weights = weights)#'IMAGENET1K_V2') # finetuning
        # self.resnet.conv1 = nn.Conv2d(num_channels, 64, 7, stride=2, padding=3, bias=False)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.out = nn.Linear(512, out_dim)

    def forward(self, x):
        """
        """
        # print(x.shape)
        z = self.feature_extractor(x).squeeze((-2, -1))
        # print(z.shape)
        z = self.out(z)
        z = F.normalize(z, dim = 1)
        return z