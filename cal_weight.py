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

import math

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

mytransform = transforms.Compose([
    transforms.ToTensor(),            # mmb
    ]
)
set = myImageFloder(root = "./data/PETA dataset",label = "traindata.txt" ,transform = mytransform)
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size = 1, shuffle = True, num_workers = 2)


print len(set)

count = 0

dataiter = iter(imgLoader)
P  = [0.0] * 35
N  = [0.0] * 35
while count < 9500:
    images,labels = dataiter.next()
    for i in range(35):         
            if labels[0][i] == 1:
                P[i] = P[i] + 1
            else :

                N[i] = N[i] + 1   
    count = count + 1



print(P)
print(N)

w = []
for i in range(35):
    print(class_names[i]+' '+str(P[i]))
    w.append(math.pow(math.e,N[i]/9500.0))

print(w)