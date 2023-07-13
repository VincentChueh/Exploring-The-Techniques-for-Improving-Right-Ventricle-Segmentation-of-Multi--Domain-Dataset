import numpy as np
import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torchio as tio

class MyDataset(Dataset):
    def __init__(self,image_dir,mask_dir,train_mode=True):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transforms = tio.Compose([
            tio.Resample((1,1,1),include=('image','mask')),  # optional step to ensure consistent voxel size
            tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, include=('image', 'mask')) # elastic deformation
        ])
        self.train_mode=train_mode
        self.masks=os.listdir(mask_dir)
        

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        img_path=os.path.join(self.image_dir,self.masks[index].replace("_gt.nii.gz",".nii.gz"))
        img=nib.load(img_path)
        image=np.array(img.dataobj)
        image=image.astype(np.float32)
        msk=nib.load(mask_path) 
        mask=np.array(msk.dataobj)
        mask=mask.astype(np.float32)
        image=torch.from_numpy(image) 
        mask=((torch.from_numpy(mask).permute(2,0,1).unsqueeze(dim=0).to(dtype=torch.long))==3).to(int) 
        mask=mask.squeeze(dim=0)
        
        m = nn.ConstantPad3d((0, 320-mask.shape[2], 0, 320-mask.shape[1], 0, 8-mask.shape[0]), 0)
        i=nn.ConstantPad3d((0,8-image.shape[3],0,320-image.shape[2],0,320-image.shape[1]),0)
        mask=m(mask)
        image=i(image)
        
        
        if self.train_mode==True:
            mask=mask.unsqueeze(dim=0)
            transformed = self.transforms({'image': image, 'mask': mask})
            image = transformed['image']
            mask = transformed['mask']
            mask=mask.squeeze(dim=0).to(torch.long)
        
        image=image.permute(0,3,1,2)
        image=torch.add(image,-image.min())
        image=torch.div(image, image.max())
        return image,mask