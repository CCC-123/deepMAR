import os
import torch
import torch.utils.data as data
from PIL import Image

import matplotlib.pyplot as plt
import torch.utils
import torchvision

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
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader):
        
        
        imgs=[]
        dirs = os.listdir(root)
        
        for dir in dirs:
            count = 0
            if os.path.isdir(root + "/" + dir):
                print(dir)
                fh = open(root + "/" + dir + "/" + 'archive' + '/' + "Label.txt")
                for line in  fh.readlines():                  
                        cls = line.split() 
                        fn = cls.pop(0)
                        att_vector=[]
                        
                        for att in class_names:
                            if att in cls:
                                att_vector.append(1)
                            else:
                                att_vector.append(0)
                        if dir == '3DPeS' or dir == 'TownCentre':                                   #sbname
                            for filename in os.listdir(root + '/' + dir + '/' +  'archive'):
                                if filename.startswith(fn+'_'):
                                    att_vector.append(os.path.join(dir + '/'  +'archive', filename))
                                    imgs.append((os.path.join(dir + '/'  +'archive', filename), att_vector))
                                    count = count + 1
                            continue

                        for filename in os.listdir(root + '/' + dir + '/' +  'archive'):
                            if filename.startswith(fn):
                                att_vector.append(os.path.join(dir + '/'  +'archive', filename))
                                imgs.append((os.path.join(dir + '/'  +'archive', filename), att_vector))
                                count = count + 1
            print(count)        

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
        return img, label

    def __len__(self):
        return len(self.imgs)
    
    def path(self):
        return 





def imshow(imgs):
    grid = torchvision.utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title("bat")
    plt.show()

mytransform = transforms.Compose([
    
    transforms.CenterCrop(224),
    transforms.ToTensor(),            # mmb
    ]
)

# torch.utils.data.DataLoader
set = myImageFloder(root = "./data/PETA dataset", transform = mytransform )
imgLoader = torch.utils.data.DataLoader(
         set, 
         batch_size = 1, shuffle = True, num_workers = 2)


print len(set)


'''dataiter = iter(imgLoader)
images,labels = dataiter.next()
print(labels)
imshow(images)'''

train = open("traindata.txt","w")
val = open("valdata.txt","w")
test = open("testdata.txt","w")
for i, data in enumerate(imgLoader, 0):
    if i < 9500:
        # get the inputs
        inputs, labels = data

        train.write(labels[35][0]+' ')

        for j in range(35):
            train.write(str(labels[j][0])+' ')
        train.write('\n')
    elif  i < 11400:
        inputs, labels = data
        val.write(labels[35][0]+' ')
        for j in range(35):
            val.write(str(labels[j][0])+' ')
        val.write('\n')
    else :
        inputs, labels = data
        test.write(labels[35][0]+' ')
        for j in range(35):
            test.write(str(labels[j][0])+' ')
        test.write('\n')
train.close()
val.close()
test.close()