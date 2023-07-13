import numpy as np
import torch
import nibabel as nib
import os
import torch
from dataset import MyDataset
from FDA import FDA_source_to_target_np

#dataset
'''
for accessing the dataset, please refer to the Dataset folder from the link: https://drive.google.com/drive/folders/1XwK5zsf27TgHs72HoKad6p7S9OGKSUD-?usp=sharing
'''
train_ds=MyDataset("/Users/vincentc/Desktop/MnM2/train_sa","Dataset/3D_SA_dataset/train_sa_mask",train_mode=False)
val_ds=MyDataset("/Users/vincentc/Desktop/MnM2/val_sa","Dataset/3D_SA_dataset/val_sa_mask",train_mode=False)
test_ds=MyDataset("/Users/vincentc/Desktop/MnM2/test_sa","Dataset/3D_SA_dataset/test_sa_mask",train_mode=False)

#image selected as the target domain 
example_file = os.path.join("/Users/vincentc/Desktop/MnM2/train_la", "141_LA_ED.nii.gz")
img = nib.load(example_file)
imgnp=torch.from_numpy(np.array(img.dataobj)).to(torch.float32)[:320,:320,:].permute(2,0,1)#288,288,1
imgnp=torch.div(imgnp,imgnp.max())

for img,msk, aff, name in train_ds:
    img=img.permute(1,0,2,3)#8,1,320,320
    for i in range(8):
        sts=torch.from_numpy(FDA_source_to_target_np(img[i],imgnp,L=0.001)).to(torch.float16)#2,320,320
        img[i]=sts
    img=img.numpy()
    img_to_save=nib.spatialimages.SpatialImage(img,aff)
    f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/3D_SA/train_sa_fda',name)
    nib.loadsave.save(img_to_save,f_path)

for img,msk, aff, name in val_ds:
    img=img.permute(1,0,2,3)#8,2,320,320
    for i in range(8):
        sts=torch.from_numpy(FDA_source_to_target_np(img[i],imgnp,L=0.001)).to(torch.float16)#2,320,320
        img[i]=sts
    img=img.numpy()
    img_to_save=nib.spatialimages.SpatialImage(img,aff)
    f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/3D_SA/val_sa_fda',name)
    nib.loadsave.save(img_to_save,f_path)

for img,msk, aff, name in test_ds:
    img=img.permute(1,0,2,3)#8,2,320,320
    for i in range(8):
        sts=torch.from_numpy(FDA_source_to_target_np(img[i],imgnp,L=0.001)).to(torch.float16)#2,320,320
        img[i]=sts
    img=img.numpy()
    img_to_save=nib.spatialimages.SpatialImage(img,aff)
    f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/3D_SA/test_sa_fda',name)
    nib.loadsave.save(img_to_save,f_path)