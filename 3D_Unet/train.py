import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import sys
from dataset import MyDataset
from model import UNet
from device import get_default_device
from evaluation import dice_coefficient, HausdorffDistance

#import dataset
train_batch_size=6
val_batch_size=4

train_ds=MyDataset("path of training image","path of training mask",train_mode=True)
val_ds=MyDataset("path of validation image","path of validation mask",train_mode=False)
test_ds=MyDataset("path of test image","path of test mask",train_mode=False)

#put into dataloader
train_dl=DataLoader(train_ds,train_batch_size,num_workers=3,shuffle=True,pin_memory=True)
val_dl=DataLoader(val_ds,val_batch_size,num_workers=3,pin_memory=True)
test_dl=DataLoader(test_ds,val_batch_size,num_workers=3,pin_memory=True)

#model 
model=nn.DataParallel(UNet(2,2))
#model=torch.load("path of pretrained model") #if using pretrained model, load model parameters

#train with device (gpu or cpu)
device=get_default_device()
model.to(device)

#evaluation object
hd=HausdorffDistance()

#training
def train(epochs,max_lr,model,train_loader,val_loader,weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    print('begin')
    optimizer=opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_loader))
    history=[]
    max_c=0
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        val_losses=[]
        train_acc=[]
        val_acc=[]
        train_hd=[]
        val_hd=[]

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
            count=0
            for i in range(train_batch_size):
                dice_score=dice_coefficient(out, masks,smooth=1e-10, n_classes=2)
                count+=dice_score
            train_acc.append(count/train_batch_size)
            t=torch.argmax(out,dim=1).unsqueeze(dim=1)
            masks=masks.unsqueeze(dim=1)
            hausdorff=hd.compute(t,masks)
            masks=masks.squeeze(dim=1)
            train_hd.append(hausdorff.item())

        with torch.no_grad():
            for images,masks in val_loader:
                images=images.to(device)
                masks=masks.to(device)
                out=model(images) #1,10,8,224,224
                loss=F.cross_entropy(out,masks)
                val_losses.append(loss.item())
                count=0
                for i in range(val_batch_size):
                    dice_score=dice_coefficient(out, masks,smooth=1e-10, n_classes=2)
                    count+=dice_score
                val_acc.append(count/val_batch_size)
                t=torch.argmax(out,dim=1).unsqueeze(dim=1)
                masks=masks.unsqueeze(dim=1)
                hausdorff=hd.compute(t,masks)
                masks=masks.squeeze(dim=1)
                val_hd.append(hausdorff.item())


        result=dict()
        result['epoch']=epoch+1
        result['train_loss']=np.mean(train_losses).item()
        result['val_loss']=np.mean(val_losses).item()
        result['train_acc']=np.mean(train_acc).item()
        result['val_acc']=np.mean(val_acc).item()
        result['train_hd']=np.mean(train_hd).item()
        result['val_hd']=np.mean(val_hd).item()

        print('epoch',epoch)
        print('train_loss',result['train_loss'])
        print('val_loss',result['val_loss'])
        print('train_acc',result['train_acc'])
        print('val_acc',result['val_acc'])
        print('train_hd',result['train_hd'])
        print('val_hd',result['val_hd'])
        history.append(result)

        #save model
        criteria=result['val_acc']
        if max_c<criteria:
            max_c=criteria
            torch.save(model,'./3d_ce500.pt')
            print('model saved! Criteria=',criteria)
        sys.stdout.flush()

    return history

x=train(500,0.01,model,train_dl,val_dl,weight_decay=1e-4, grad_clip=0.1,opt_func=torch.optim.Adam)

