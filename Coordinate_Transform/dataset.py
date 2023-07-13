import numpy as np
import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms.functional as fn
from nibabel.affines import apply_affine
from numpy.linalg import inv

class MyDataset(Dataset):
    def __init__(self,la_image_dir,sa_image_dir,la_mask_dir,sa_mask_dir,la_transform=None,sa_transform=None,train_mode=True):
        self.la_image_dir=la_image_dir
        self.sa_image_dir=sa_image_dir
        self.la_mask_dir=la_mask_dir
        self.sa_mask_dir=sa_mask_dir
        self.la_transform=la_transform
        self.sa_transform=sa_transform
        self.train_mode=train_mode
        self.sa_masks=os.listdir(sa_mask_dir)
        self.la_masks=os.listdir(la_mask_dir)

    def __len__(self):
        return len(self.la_masks)

    def __getitem__(self, index):
        la_mask_path=os.path.join(self.la_mask_dir,self.la_masks[index])
        la_img_path=os.path.join(self.la_image_dir,self.la_masks[index].replace("_gt.nii.gz",".nii.gz"))
        sa_mask_path=os.path.join(self.sa_mask_dir,self.la_masks[index].replace("_LA_","_SA_"))
        sa_img_path=os.path.join(self.sa_image_dir,self.la_masks[index].replace("_gt.nii.gz",".nii.gz")).replace("_LA_","_SA_")
        sa_image_name=os.path.basename(sa_img_path)
        la_img=nib.load(la_img_path)
        sa_img=nib.load(sa_img_path)
        la_image=np.array(la_img.dataobj)
        sa_image=np.array(sa_img.dataobj)
        la_affine=la_img.affine
        sa_affine=inv(sa_img.affine)
        la_image=la_image.astype(np.float32)
        sa_image=sa_image.astype(np.float32)
        sa_image=torch.from_numpy(sa_image).permute(1,0,2,3)
        
        la_map=dict()
        for i in range (488):
            for j in range(488):
                coord=(i,j,0)
                com_coord=apply_affine(la_affine,coord)
                sa_coord=apply_affine(sa_affine,com_coord).astype(int)
                la_map[(i,j)]=sa_coord

        la_image=torch.from_numpy(la_image).permute(2,0,1)


        tr_si= nn.ConstantPad3d((0, 0, 0, 0, 0, 8-sa_image.shape[1]), 0)
        if sa_image.shape[2]<=320:
            sa_image=fn.pad(sa_image,(0,0,0,320-sa_image.shape[2]),0)
        else:
            sa_image=sa_image[:,:,:320,:]
        if sa_image.shape[3]<=320:
            sa_image=fn.pad(sa_image,(0,0,320-sa_image.shape[3],0),0)
        else:
            sa_image=sa_image[:,:,:,:320]
        if sa_image.shape[1]<8:
            sa_image=tr_si(sa_image)
        else:
            sa_image=sa_image[:,:8,:,:]

        
        if la_image.shape[1]<=448:
            la_image=fn.pad(la_image,(0,0,0,448-la_image.shape[1]),0)
        else:
            la_image=la_image[:,:448,:]
        if la_image.shape[2]<=448:
            la_image=fn.pad(la_image,(0,0,448-la_image.shape[2],0),0)
        else:
            la_image=la_image[:,:,:448]
        
        return la_image,sa_image,la_affine,sa_affine,la_map,sa_image_name