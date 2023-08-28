import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
import random
import os
import shutil
from eval import * 

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                    nn.Conv2d(in_channels=1+1,out_channels=16,kernel_size=(7,7),stride=(1,1),padding=3,bias=False,padding_mode='reflect'),
                    nn.PReLU(num_parameters=1,init=0.25),
                    nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,padding_mode='reflect'),
                    nn.PReLU(num_parameters=1,init=0.25),
                    nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,padding_mode='reflect'),
                    nn.PReLU(num_parameters=1,init=0.25),
                    nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,padding_mode='reflect'),
                )

    def forward(self,x,p,q):
        x_ = torch.cat([x,p],dim=1)
        y =  q + self.model(x_)
        return y

    def initialize(self,type = 'normal',val = None):
        for idx,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,mean = 0, std = 1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias,val=0.0)

class Trainer(nn.Module):
    def __init__(self,model,device,band_id,recoder):
        super().__init__()
        self.model = model.to(device)
        if os.path.exists('./model/{}/{}/net.pth'.format(band_id,recoder)):
            self.model.load_state_dict(torch.load('./model/{}/{}/net.pth'.format(band_id,recoder)))
        else:
            self.model.initialize()
            pass
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-3,weight_decay=0.2)        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.2)
        self.device = device
        self.L1Loss = nn.L1Loss().to(device)
        self.L2Loss = nn.MSELoss().to(device)

    def forward(self,X,P,D,L,M,Q_hr,Q_lr):
        return self.loss_fn(X,P,D,L,M,Q_hr,Q_lr)

    def loss_fn(self,X,P,D,L,M,Q_hr,Q_lr):
 
        Xg = self.model(X.to(self.device),P.to(self.device),Q_hr.to(self.device))
        content_full_loss = self.L2Loss(Xg.to(self.device),Q_hr.to(self.device)) 
        Xr = self.model(D.to(self.device),L.to(self.device),Q_lr.to(self.device))
        content_reduced_loss = self.L2Loss(Xr.to(self.device),M.to(self.device))

        if content_full_loss < content_reduced_loss:
            scale_difference_loss = 0
        else:
            scale_difference_loss = self.L2Loss(content_full_loss,content_reduced_loss)

        loss = 1 * content_full_loss  + 1 * content_reduced_loss  + 2 * scale_difference_loss
        return loss

    def img2tf(self,image_np):
        if len(image_np.shape) == 2:
            image_tf = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
        else:
            image_tf = torch.from_numpy(image_np).permute(2,0,1).unsqueeze(0)
        
        return image_tf

    def tf2img(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0)
        
        return image_np


def predict_spatial_info(X_np,P_np,D_np,L_np,Q_hr_np,Q_lr_np,M_gt_np,device,recoder = None,band_id = None):
    model = network()
    trainer = Trainer(model,device,band_id,recoder)
    X = trainer.img2tf(X_np.astype(np.float32))
    P = trainer.img2tf(P_np.astype(np.float32))
    D = trainer.img2tf(D_np.astype(np.float32))
    L = trainer.img2tf(L_np.astype(np.float32))
    M = trainer.img2tf(M_gt_np.astype(np.float32))
    Q_hr = trainer.img2tf(Q_hr_np.astype(np.float32))
    Q_lr = trainer.img2tf(Q_lr_np.astype(np.float32))
    trainer.to(device)
    trainer.train()
    if os.path.exists('./model/{}/{}/net.pth'.format(band_id,recoder)):
        n_epochs,loss_maximum = 50,999
    else:
        os.makedirs('./model/{}/{}/'.format(band_id,recoder))
        n_epochs,loss_maximum = 500,999
    n_epochs_bar = tqdm(range(n_epochs)) 
    for epoch in n_epochs_bar:
        trainer.optimizer.zero_grad()
        loss = trainer(X,P,D,L,M,Q_hr,Q_lr)
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()

        with torch.no_grad():
            Y = trainer.model(D.to(device,dtype=torch.float),L.to(device,dtype=torch.float),Q_lr.to(device,dtype=torch.float))
            Z = trainer.tf2img(Y)
            rmse_lr_gd = RMSE(Z,Q_lr_np) * 255
            rmse_lr = RMSE(Z,M_gt_np) * 255

        with torch.no_grad():
            Y = trainer.model(X.to(device,dtype=torch.float),P.to(device,dtype=torch.float),Q_hr.to(device,dtype=torch.float))
            Z = trainer.tf2img(Y)
            rmse_hr_gd = RMSE(Z,Q_hr_np) * 255

        n_epochs_bar.set_description('Epoch : {}/{} rmse_lr : {:.3f} rmse_gd : {:.3f} rmse_gd : {:.3f} Loss : {:.6f}'.format(epoch,n_epochs,rmse_lr,rmse_lr_gd,rmse_hr_gd,loss.item()))
 
        if loss_maximum > loss.item() :
            loss_maximum = loss.item()
            torch.save(trainer.model.state_dict(),'./model/{}/{}/net.pth'.format(band_id,recoder))
            pass
        pass

    trainer.model.load_state_dict(torch.load('./model/{}/{}/net.pth'.format(band_id,recoder)))
    trainer.eval()
    Xg = trainer.model(X.to(device,dtype=torch.float),P.to(device,dtype=torch.float),Q_hr.to(device))
    Xg = trainer.tf2img(Xg)
    return np.clip(Xg,-1,1)

if __name__ == '__main__':
    print('Hello World')