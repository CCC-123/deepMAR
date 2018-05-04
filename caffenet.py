import torch.nn as nn
import torch.nn.modules.normalization as norm

class CaffeNet(nn.Module):

    def __init__(self, num_classes=35):
        super(CaffeNet, self).__init__()
        self.features = nn.Sequential(
            #227x227x3
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            #55x55x96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #27 x 27 x 96
            norm.CrossMapLRN2d(5),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            #27 x 27 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #13 x 13 x 256
            norm.CrossMapLRN2d(5),


            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            # 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            # 13 x 13 x 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            #35
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

