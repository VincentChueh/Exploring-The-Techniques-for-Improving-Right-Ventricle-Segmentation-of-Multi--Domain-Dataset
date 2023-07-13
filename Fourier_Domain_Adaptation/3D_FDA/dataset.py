import numpy as np
import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms.functional as fn

class MyDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None,train_mode=False):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.train_mode=train_mode
        self.masks=os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        img_path=os.path.join(self.image_dir,self.masks[index].replace("_gt.nii.gz",".nii.gz"))
        name=os.path.basename(img_path)
        img=nib.load(img_path)
        affine=img.affine
        image=np.array(img.dataobj)
        image=image.astype(np.float32)
        msk=nib.load(mask_path) 
        mask=np.array(msk.dataobj)
        mask=mask.astype(np.float32)
        image=torch.from_numpy(image).unsqueeze(dim=0).permute(0,3,1,2)#z,c,w,h
        mask=torch.from_numpy(mask).permute(2,0,1).unsqueeze(dim=0).to(dtype=torch.long) #12,224,224

        
        mask=mask.squeeze(dim=0)
        tr_sm= nn.ConstantPad3d((0, 0, 0, 0, 0, 8-mask.shape[0]), 0)
        tr_si= nn.ConstantPad3d((0, 0, 0, 0, 0, 8-image.shape[1]), 0)

        
        if image.shape[2]<=320:
            image=fn.pad(image,(0,0,0,320-image.shape[2]),0)
        else:
            image=image[:,:,:320,:]
        if image.shape[3]<=320:
            image=fn.pad(image,(0,0,320-image.shape[3],0),0)
        else:
            image=image[:,:,:,:320]
        if image.shape[1]<8:
            image=tr_si(image)
        else:
            image=image[:,:8,:,:]

        if mask.shape[1]<=320:
            mask=fn.pad(mask,(0,0,0,320-mask.shape[1]),0)
        else:
            mask=mask[:,:320,:]
        if mask.shape[2]<=320:
            mask=fn.pad(mask,(0,0,320-mask.shape[2],0),0)
        else:
            mask=mask[:,:,:320]
        if mask.shape[0]<8:
            mask=tr_sm(mask)
        else:
            mask=mask[:8,:,:]
        
        image=torch.div(image,image.max())
        return image,mask,affine,name