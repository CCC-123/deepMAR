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

import sys


class_names = ['personalLess30','personalLess45','personalLess60','personalLarger60','carryingBackpack',
                'carryingOther','lowerBodyCasual','upperBodyCasual','lowerBodyFormal','upperBodyFormal',
                'accessoryHat','upperBodyJacket','lowerBodyJeans','footwearLeatherShoes','upperBodyLogo',
                'hairLong','personalMale','carryingMessengerBag','accessoryMuffler','accessoryNothing',
                'carryingNothing','upperBodyPlaid','carryingPlasticBags','footwearSandals','footwearShoes',
                'lowerBodyShorts','upperBodyShortSleeve','lowerBodyShortSkirt','footwearSneakers','upperBodyThinStripes', #longskirt,thickstripe
                'accessorySunglasses','lowerBodyTrousers','upperBodyTshirt','upperBodyOther','upperBodyVNeck']
class_len = 35





def Judge(imgpath) :
        mytransform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),            # mmb
            ]
        )

        path = "/home/ubuntu/Desktop/deepMAR/checkpoint3/checkpoint_epoch_360" 
        net = models.alexnet(num_classes=35)
        net.load_state_dict(torch.load(path))
        net.eval()
        net.cuda()

        images = Image.open(imgpath).convert('RGB')
        images = mytransform(images)
        images = images.view(1,3,227,227)
        inputs = Variable(images,volatile = True).cuda()
        outputs = net(inputs)
            
        count = 0
        ret = []
        for item in outputs[0]:
            if item.data[0] > 0:
                ret.append(class_names[count])
            count = count + 1
        return ret
















