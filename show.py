import os
import torch
import torch.utils.data as data
from PIL import Image

import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.models as models


import scipy.io as scio
import torchvision.transforms as transforms



class_names = ['personalLess30','personalLess45','personalLess60','personalLarger60','carryingBackpack',
                'carryingOther','lowerBodyCasual','upperBodyCasual','lowerBodyFormal','upperBodyFormal',
                'accessoryHat','upperBodyJacket','lowerBodyJeans','footwearLeatherShoes','upperBodyLogo',
                'hairLong','personalMale','carryingMessengerBag','accessoryMuffler','accessoryNothing',
                'carryingNothing','upperBodyPlaid','carryingPlasticBags','footwearSandals','footwearShoes',
                'lowerBodyShorts','upperBodyShortSleeve','lowerBodyShortSkirt','footwearSneakers','upperBodyThinStripes', #longskirt,thickstripe
                'accessorySunglasses','lowerBodyTrousers','upperBodyTshirt','upperBodyOther','upperBodyVNeck']
class_len = 35


def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFloder(data.Dataset):
    def __init__(self, root, label,transform = None, target_transform = None, loader = default_loader):
        
        
        imgs=[]
        
        
        file = open(label)
        for line in file.readlines():
            cls = line.split()
            pos = cls.pop(0)
            att_vector = []
            for att in cls:
                if att == '0':
                    att_vector.append(0)
                else:
                    att_vector.append(1)
            if os.path.isfile(os.path.join(root,pos)):
                #imgs.append((name[0][0],[x*2-1 for x in testlabel[count]]))   (-1,1)
                imgs.append((pos,att_vector))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root,fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)
    

def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("bat")
    plt.show()

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    path = "./checkpoint/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(),path)





mytransform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),            # mmb
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PETA dataset",label = "testdata.txt", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size = 1, shuffle = True, num_workers = 2)


print len(set)


path = "./checkpoint3/checkpoint_epoch_360" 
net = models.alexnet(num_classes=35)
net.load_state_dict(torch.load(path))
net.cuda()


'''weight = torch.ones(1,26)
weight[0][2] = 0.3       #18-60
weight[0][22] = 0.4      #trousers
criterion = nn.BCEWithLogitsLoss(weight = weight)          #TODO:1.learn 2. weight'''

dataiter = iter(imgLoader)




x = " "
while True and x != 'n':
    images,labels = dataiter.next()
    inputs, labels = Variable(images,volatile = True).cuda(), Variable(labels).cuda()
    outputs = net(inputs)
    



    print(outputs)
    print(labels)

    count = 0

    for item in outputs[0]:

            if item.data[0] > 0:
                print(class_names[count])
            count = count + 1
    print('\n')

    count = 0
    for item in labels[0]:
        if item.data[0] > 0:
            print(class_names[count])
        count = count + 1
    imshow(images)   
    x = raw_input()













