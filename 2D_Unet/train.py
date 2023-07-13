import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from dataset import RV
from augmentation import train_ag, val_ag
from model import UNet
from device import get_default_device
from evaluation import dice_coefficient, HausdorffDistance

#import dataset
train_ds=RV("image file path","mask file path",transform=train_ag)
val_ds=RV("image file path","mask file path",transform=val_ag)
test_ds=RV("image file path","mask file path",transform=val_ag)

#put into dataLoader
train_batch_size=6
val_batch_size=6
train_dl=DataLoader(train_ds,train_batch_size,num_workers=3,shuffle=True,pin_memory=True)
val_dl=DataLoader(val_ds,val_batch_size,num_workers=3,pin_memory=True)
test_dl=DataLoader(test_ds,val_batch_size,num_workers=3,pin_memory=True)

#model
model=UNet(1,2)

#put model on device
device=get_default_device()
model.to(device)

#hausdorff distance object
hd=HausdorffDistance()

def train(epochs,max_lr,model,train_loader,val_loader,weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history=[]
    optimizer=opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_loader))
    max_c=1000
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        val_losses=[]
        train_hd=[]
        val_hd=[]
        train_dice_coefficient=[]
        val_dice_coefficient=[]

        for images,masks in train_loader:
            images=images.to(device)
            masks=masks.to(device)
            out=model(images)  
            loss=F.cross_entropy(out,masks)
            loss.backward()
            train_losses.append(loss.item())
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            d=dice_coefficient(out,masks)
            train_dice_coefficient.append(d)
            t=torch.argmax(out,dim=1).unsqueeze(dim=1)
            masks=masks.unsqueeze(dim=1)
            d=hd.compute(t,masks)
            masks=masks.squeeze(dim=1)
            train_hd.append(d.item())

        with torch.no_grad():
            for images,masks in val_loader:
                images=images.to(device)
                masks=masks.to(device)
                out=model(images)
                loss=F.cross_entropy(out,masks)
                val_losses.append(loss.item())
                y=dice_coefficient(out,masks)
                val_dice_coefficient.append(y)
                t=torch.argmax(out,dim=1).unsqueeze(dim=1)
                masks=masks.unsqueeze(dim=1)
                d=hd.compute(t,masks)
                masks=masks.squeeze(dim=1)
                val_hd.append(d.item())

        result=dict()
        result['epoch']=epoch+1
        result['train_loss']=np.mean(train_losses).item()
        result['val_loss']=np.mean(val_losses).item()
        result['train_dice_coefficient']=np.mean(train_dice_coefficient).item()
        result['val_dice_coefficient']=np.mean(val_dice_coefficient).item()
        result['train_hd']=np.mean(train_hd).item()
        result['val_hd']=np.mean(val_hd).item()

        print('epoch',epoch)
        print('train_loss',result['train_loss'])
        print('val_loss',result['val_loss'])
        print('train_dice_coefficient',result['train_dice_coefficient'])
        print('val_dice_coefficient',result['val_dice_coefficient'])
        print('train_hd',result['train_hd'])
        print('val_hd',result['val_hd'])
        history.append(result)
        
        #save model
        criteria=(result['val_hd'])
        if max_c> criteria :
            max_c=criteria
            torch.save(model,'./ce500.pt')
            print('model saved!',criteria)
    return history
x=train(100,0.01,model,train_dl,val_dl,weight_decay=1e-4, grad_clip=0.1,opt_func=torch.optim.Adam)


