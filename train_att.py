import os
import torch
import torch.utils.data as data
from PIL import Image

import att

import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.utils
import torchvision
import torchvision.models as models


import scipy.io as scio
import torchvision.transforms as transforms


from visdom import Visdom
import numpy as np
viz = Visdom()
win = viz.line(
    Y=np.array([0.5]),
    name="2"
)

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
    if not os.path.exists("checkpoint5"):
        os.mkdir("checkpoint5")
    path = "./checkpoint5/checkpoint_epoch_{}".format(epoch)
    torch.save(net.state_dict(),path)





mytransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    #transforms.RandomCrop(227),
    transforms.ToTensor(),            # mmb
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PETA dataset",label = "traindata.txt", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size = 1, shuffle = True, num_workers = 2)


print len(set)



net = att.CaffeNet()


net_dict = net.state_dict()
path = "./checkpoint1/checkpoint_epoch_360" 
pretrained_dict = torch.load(path)
pretrained_dict = {k : v for k,v in pretrained_dict.items() if k in net_dict and pretrained_dict[k].size() == net_dict[k].size()}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict) 


net.train()
net.cuda()


weight = torch.Tensor([1.6577705942628165, 1.9569133706913606, 2.453397219303324, 2.5511042784138787, 
2.234931382642667, 2.2346961390891087, 1.1512428584451406, 1.1603675891347502, 
2.3646536876713378, 2.3726322669984423, 2.455205651858246, 2.5371784866683735, 
2.0040254740364105, 2.020547311604474, 2.6100475016963145, 2.1506916190382617, 
1.566662199124304, 2.0233141637611203, 2.505331478757186, 1.2855130440514353, 
2.0552984172896176, 2.652144136746138, 2.5188173194587176, 2.66782399742227, 
1.8903024488309481, 2.6218881780191023, 2.35422250465764, 2.6015443856359055, 
2.181013056518632, 2.6723209605343974, 2.6365564328356172, 1.622039341816392, 
2.4982211718819127,1.717271754249473, 2.6861401189973217])
criterion = nn.BCEWithLogitsLoss(weight = weight)
criterion.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001,momentum=0.9 )

running_loss = 0.0
for epoch in range(1000):
    for i, data in enumerate(imgLoader, 0):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            
            #print(outputs)

            loss = criterion(outputs, labels) 


            loss.backward()        
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 0: # print every 2000 mini-batches
                print('[ %d %5d] loss: %.3f' % ( epoch,i+1, running_loss / 100))
                viz.updateTrace(
                    X=np.array([epoch+i/1900.0]),
                    Y=np.array([running_loss]),
                    win=win,
                    name="2"
                )
                running_loss = 0.0
    if epoch % 30 == 0:
        checkpoint(epoch)














