import torch
import nibabel as nib
import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler
from nibabel.affines import apply_affine
from numpy.linalg import inv
from device import get_default_device
from dataset import MyDataset
from augmentation import train_ag,val_ag
from model import UNet


#input dataset
train_ds=MyDataset("/Users/vincentc/Desktop/MnM2/train_la","/Users/vincentc/Desktop/FDA_Dataset/3D_SA/train_sa_fda","/Users/vincentc/Desktop/MnM2/train_la_mask","/Users/vincentc/Desktop/MnM2/train_sa_mask",la_transform=train_ag,train_mode=True)
val_ds=MyDataset("/Users/vincentc/Desktop/MnM2/val_la","/Users/vincentc/Desktop/FDA_Dataset/3D_SA/val_sa_fda","/Users/vincentc/Desktop/MnM2/val_la_mask","/Users/vincentc/Desktop/MnM2/val_sa_mask",la_transform=val_ag,train_mode=False)
test_ds=MyDataset("/Users/vincentc/Desktop/MnM2/test_la","/Users/vincentc/Desktop/FDA_Dataset/3D_SA/test_sa_fda","/Users/vincentc/Desktop/MnM2/test_la_mask","/Users/vincentc/Desktop/MnM2/test_sa_mask",la_transform=val_ag,train_mode=False)


#DataLoader
batch_size=10
train_dl=DataLoader(train_ds,batch_size,shuffle=True,pin_memory=True)
val_dl=DataLoader(val_ds,batch_size,pin_memory=True)
test_dl=DataLoader(test_ds,batch_size,pin_memory=True)

#load the trained 2D Unet
model=torch.load("ce500.pt")

#use GPU
device=get_default_device()
model.to(device)
model.train()

for la_image,sa_image,la_affine, sa_affine,la_map,name in train_dl:
    print(la_image.shape)#10,1,448,448
    print(sa_image.shape)#10,1,8,320,320
    out2d=model(la_image)           
    out2d=torch.argmax(F.softmax(out2d.permute(0,2,3,1),dim=3),dim=3).unsqueeze(dim=3)#10,448,448,1
    sa_image1=torch.zeros((sa_image.shape[0],320,320,8,1))
    for x in range(sa_image.shape[0]):
        for i in range(448):
            for j in range(448):
                if out2d[x][i,j,0]!=0:
                    sa_coord=la_map[(i,j)][x]
                    #print('sa_coord::',sa_coord.shape)
                    if sa_coord[0]<320 and sa_coord[0]>=0 and sa_coord[1]<320 and sa_coord[1]>=0 and sa_coord[2]<8 and sa_coord[2]>=0:
                        sa_image1[x][sa_coord[0]][sa_coord[1]][sa_coord[2]][0]=out2d[x][i,j,0]
                    else:
                        pass
    sa_image1=sa_image1.permute(0,4,3,1,2)#10,1,8,224,224  
    sa_image1=torch.div(sa_image1,sa_image1.max())
    sa_image=torch.div(sa_image,sa_image.max())
    sa_image=torch.cat((sa_image,sa_image1),dim=1).permute(0,1,3,4,2)#3,2,320,320,8
    print(sa_image.max())
    print(sa_image.min())
    
    #sa_image=sa_image.permute(0,4,1,2,3)
    for x in range(sa_image.shape[0]):
        '''
        for i in range(8):
            img=tt.ToPILImage()(sa_image[x][i][:1,:,:])
            img.show()
            img=tt.ToPILImage()(sa_image[x][i][1:2,:,:])
            img.show()
        break
        '''
        img=sa_image.clone().detach()
        aff=sa_affine.clone().detach()
        img=img[x].numpy()
        aff=aff[x].numpy()
        img_to_save=nib.spatialimages.SpatialImage(img,aff)
        f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/concat_dataset/train_2channel',name[x])
        nib.loadsave.save(img_to_save,f_path)
        print('train_done')


for la_image,sa_image,la_affine, sa_affine,la_map,name in val_dl:
    print(la_image.shape)#10,1,448,448
    print(sa_image.shape)#10,1,8,320,320
    out2d=model(la_image)           
    out2d=torch.argmax(F.softmax(out2d.permute(0,2,3,1),dim=3),dim=3).unsqueeze(dim=3)#10,448,448,1
    sa_image1=torch.zeros((sa_image.shape[0],320,320,8,1))
    for x in range(sa_image.shape[0]):
        for i in range(448):
            for j in range(448):
                if out2d[x][i,j,0]!=0:
                    sa_coord=la_map[(i,j)][x]
                    #print('sa_coord::',sa_coord.shape)
                    if sa_coord[0]<320 and sa_coord[0]>=0 and sa_coord[1]<320 and sa_coord[1]>=0 and sa_coord[2]<8 and sa_coord[2]>=0:
                        sa_image1[x][sa_coord[0]][sa_coord[1]][sa_coord[2]][0]=out2d[x][i,j,0]
                    else:
                        pass
    sa_image1=sa_image1.permute(0,4,3,1,2)#10,1,8,224,224  
    sa_image1=torch.div(sa_image1,sa_image1.max())
    sa_image=torch.div(sa_image,sa_image.max())
    sa_image=torch.cat((sa_image,sa_image1),dim=1).permute(0,1,3,4,2)#3,2,320,320,8
    print(sa_image.max())
    print(sa_image.min())
    
    #sa_image=sa_image.permute(0,4,1,2,3)
    for x in range(sa_image.shape[0]):
        '''
        for i in range(8):
            img=tt.ToPILImage()(sa_image[x][i][:1,:,:])
            img.show()
            img=tt.ToPILImage()(sa_image[x][i][1:2,:,:])
            img.show()
        break
        '''
        img=sa_image.clone().detach()
        aff=sa_affine.clone().detach()
        img=img[x].numpy()
        aff=aff[x].numpy()
        img_to_save=nib.spatialimages.SpatialImage(img,aff)
        f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/concat_dataset/val_2channel',name[x])
        nib.loadsave.save(img_to_save,f_path)
        print('val_done')



for la_image,sa_image,la_affine, sa_affine,la_map,name in test_dl:
    print(la_image.shape)#10,1,448,448
    print(sa_image.shape)#10,1,8,320,320
    out2d=model(la_image)           
    out2d=torch.argmax(F.softmax(out2d.permute(0,2,3,1),dim=3),dim=3).unsqueeze(dim=3)#10,448,448,1
    sa_image1=torch.zeros((sa_image.shape[0],320,320,8,1))
    for x in range(sa_image.shape[0]):
        for i in range(448):
            for j in range(448):
                if out2d[x][i,j,0]!=0:
                    sa_coord=la_map[(i,j)][x]
                    #print('sa_coord::',sa_coord.shape)
                    if sa_coord[0]<320 and sa_coord[0]>=0 and sa_coord[1]<320 and sa_coord[1]>=0 and sa_coord[2]<8 and sa_coord[2]>=0:
                        sa_image1[x][sa_coord[0]][sa_coord[1]][sa_coord[2]][0]=out2d[x][i,j,0]
                    else:
                        pass
    sa_image1=sa_image1.permute(0,4,3,1,2)#10,1,8,224,224  
    sa_image1=torch.div(sa_image1,sa_image1.max())
    sa_image=torch.div(sa_image,sa_image.max())
    sa_image=torch.cat((sa_image,sa_image1),dim=1).permute(0,1,3,4,2)#3,2,320,320,8
    print(sa_image.max())
    print(sa_image.min())
    
    #sa_image=sa_image.permute(0,4,1,2,3)
    for x in range(sa_image.shape[0]):
        '''
        for i in range(8):
            img=tt.ToPILImage()(sa_image[x][i][:1,:,:])
            img.show()
            img=tt.ToPILImage()(sa_image[x][i][1:2,:,:])
            img.show()
        break
        '''
        img=sa_image.clone().detach()
        aff=sa_affine.clone().detach()
        img=img[x].numpy()
        aff=aff[x].numpy()
        img_to_save=nib.spatialimages.SpatialImage(img,aff)
        f_path=os.path.join('/Users/vincentc/Desktop/FDA_Dataset/concat_dataset/test_2channel',name[x])
        nib.loadsave.save(img_to_save,f_path)
        print('test_done')
        
