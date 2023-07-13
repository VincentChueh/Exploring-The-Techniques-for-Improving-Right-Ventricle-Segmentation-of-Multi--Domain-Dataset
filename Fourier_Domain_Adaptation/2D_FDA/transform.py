import numpy as np
import torch
import os
import nibabel as nib
from augmentation import train_ag,val_ag
from dataset import RV
from FDA import FDA_source_to_target_np

#Dataset
train_ds=RV("/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/train_la","/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/train_la_mask",transform=train_ag)
val_ds=RV("/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/val_la","/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/val_la_mask",transform=val_ag)
test_ds=RV("/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/test_la","/Users/vincentc/Desktop/psxsc15_thesis_report/Dataset/2D_LA_dataset/test_la_mask",transform=val_ag)

#select the image as the target domain
target_file = os.path.join("/Users/vincentc/Desktop/MnM2/train_la", "141_LA_ED.nii.gz")
img = nib.load(target_file)
target=torch.from_numpy(np.array(img.dataobj)).to(torch.float32)[:448,:448,:].permute(2,0,1)
target=torch.div(target,target.max())

#transform
x=0
for img, _, image_name, affine in train_ds:
    img=torch.div(img,img.max())
    sts=torch.from_numpy(FDA_source_to_target_np(img,target,L=0.005)).numpy()
    img_to_save=nib.spatialimages.SpatialImage(sts,affine)
    f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/2D_LA/t',image_name)
    nib.loadsave.save(img_to_save,f_path)
    break