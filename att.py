import torch.nn as nn
import torch.nn.modules.normalization as norm
import torch.nn.functional as F

class CaffeNet(nn.Module):

    def __init__(self, num_classes=35):
        super(CaffeNet, self).__init__()
        self.features = nn.Sequential(
            #227x227x3
            nn.Conv2d(3, 96, kernel_size=11),
            #55x55x96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            #27 x 27 x 96
            norm.CrossMapLRN2d(5),

            nn.Conv2d(96, 256, kernel_size=5),
            #27 x 27 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            #13 x 13 x 256
            norm.CrossMapLRN2d(5),


            nn.Conv2d(256, 384, kernel_size=3),
            # 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3),
            # 13 x 13 x 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),
            #13 x 13 x 256
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #6 x 6 x 256
        )
        

        #self.fcn = nn.Conv2d(256,num_classes,kernel_size = 6)
            #1 x 1 x 35
        
        self.fc = nn.Linear(256,num_classes)
    def forward(self, x):
        x = self.features(x)
        #x = self.fcn(x)

        x = F.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

