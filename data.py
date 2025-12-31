import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from logging import root
from torch.utils.data import Dataset
import cv2
import os

class mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,index):
        img_name = self.img_path[index]
        img_dir = os.path.join(self.root_dir,self.label_dir,img_name)
        img = cv2.imread(img_dir)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data\\train"
ants_label = "ants"
bees_label = "bees"
ants_dataset = mydata(root_dir,ants_label)
bees_dataset = mydata(root_dir,bees_label)
train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))
print(len(ants_dataset))
img,label = ants_dataset[0]
print(img.shape)

class dataset(Dataset):
    def __init__(self,root_dir,label_dir,transform = None):
        self.root_dir = root_dir
        self.label_dir = self.label_dir
        self.path = os.path.join(self,self.root_dir,self.label.dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform
    def __getitem__(self,index):
        img_name = self.img_path[index]
        img_dir = os.path.join(self.root_dir,self.label_dir,img_name)
        img = cv2.imread(img_dir)
        label = self.label_dir
        if self.transform:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.img_path)