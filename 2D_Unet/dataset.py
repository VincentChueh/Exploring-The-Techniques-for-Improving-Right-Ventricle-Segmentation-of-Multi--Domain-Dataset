import numpy as np
import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import albumentations as A
import albumentations.augmentations.geometric.transforms as gt



class RV(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform = transform
        self.masks=os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self,index):
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        img_path=os.path.join(self.image_dir,self.masks[index].replace("_gt.nii.gz",".nii.gz"))
        img=nib.load(img_path)
        image=np.array(img.dataobj)
        image=image.astype(np.float32)
        msk=nib.load(mask_path)
        mask=np.array(msk.dataobj)
        #mask=mask.astype(np.float32)
        mask=(mask==3).astype(np.float32)

        if self.transform is not None:
            augmentations=self.transform(image=image,mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
    
        image=torch.from_numpy(image)#.permute(2,0,1)
        if image.shape[1]<=448:
            image=fn.pad(image,(0,0,0,448-image.shape[1]),0)
        else:
            image=image[:,:448,:]
        if image.shape[2]<=448:
            image=fn.pad(image,(0,0,448-image.shape[2],0),0)
        else:
            image=image[:,:,:448]
        mask=torch.from_numpy(mask).permute(2,0,1).squeeze(dim=0).to(dtype=torch.long)
        if mask.shape[0]<=448:
            mask=fn.pad(mask,(0,0,0,448-mask.shape[0]),0)
        else:
            mask=mask[:448,:]
        if mask.shape[1]<=448:
            mask=fn.pad(mask,(0,0,448-mask.shape[1],0),0)
        else:
            mask=mask[:,:448]

    
        #image=torch.add(image,-image.min())
        image=torch.div(image, image.max())
        return image,mask