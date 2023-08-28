import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from tqdm import tqdm 
import os 
from skimage.metrics import peak_signal_noise_ratio as PSNR

class network(nn.Module):
    def __init__(self,
                 n_bands = 4,
                 radius = 4):
        super().__init__()
        self.n_bands = n_bands
        self.radius = radius
        self.conv1 = nn.Conv2d(in_channels=n_bands,out_channels=n_bands,kernel_size=(radius,radius),stride=(radius,radius),bias=True)
        self.act_fc1 = nn.PReLU(num_parameters=1,init=1)
        self.ext1 = nn.Conv2d(in_channels=n_bands,out_channels=n_bands*2,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.ext_fc1 = nn.PReLU(num_parameters=1,init=1)
        self.ext2 = nn.Conv2d(in_channels=n_bands*2,out_channels=n_bands*8,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.ext_fc2 = nn.PReLU(num_parameters=1,init=1)
        self.ext3 = nn.Conv2d(in_channels=n_bands*8,out_channels=n_bands*16,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.ext_fc3 = nn.PReLU(num_parameters=1,init=1)

    def forward(self,x):
        w_local = self.act_fc1(self.conv1(x))
        w_ex = self.ext_fc3(self.ext3(self.ext_fc2(self.ext2(self.ext_fc1(self.ext1(w_local)))))) 
        w_ext = F.pixel_shuffle(w_ex,4)        
        return w_local,w_ext

    def initialize(self,type = 'normal',val = None):
        for idx,m in enumerate(self.modules()):
            if idx == 1 and isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight,1/self.n_bands/self.radius/self.radius)
                if m.bias is not None:
                    nn.init.constant_(m.bias,val=0.0)


class Trainer(nn.Module):
    def __init__(self,model,device):
        super().__init__()
        self.model = model.to(device)
        if os.path.exists('./model/rnet.pth'):
            self.model.load_state_dict(torch.load('./model/rnet.pth'))
            self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-3)     
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-3)  
            self.model.initialize()
            pass
        self.model.to(device)      
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.2)
        self.device = device
        self.L1Loss = nn.L1Loss().to(device)
        self.L2Loss = nn.MSELoss().to(device)

    def forward(self,X,P,M,L):
        return self.loss_fn(X,P,M,L)

    def loss_fn(self,X,P,M,L):
        """
            Before training, X and P should be transfer to batch(Use pixel2batch_shuffle)
        """
        w_local,w_ext = self.model(X.to(self.device))
        cube_lr = w_local*M.to(self.device)
        cube_hr = w_ext*X.to(self.device)
        cube_lr_c = cube_lr.clone()
        cube_hr_c = cube_hr.clone()
        w_local_c = w_local.clone()
        w_ext_c = w_ext.clone()
        p_lr = torch.sum(cube_lr,dim=1).unsqueeze(1)
        p_hr = torch.sum(cube_hr,dim=1).unsqueeze(1)
        loss_content_lr = self.L1Loss(p_lr.to(self.device),L.to(self.device))
        loss_content_hr = self.L1Loss(p_hr.to(self.device),P.to(self.device))
        loss_content = 0.5 * loss_content_hr + 0.5 * loss_content_lr
        restore_lr = torch.zeros_like(M).to(self.device)
        restore_hr = torch.zeros_like(X).to(self.device)
        for i in range(4):
            channnel_list = []
            for j in range(4):
                if j != i:
                    channnel_list.append(j)
                    pass
                pass
            restore_lr[:,i,:,:] = torch.div(L.clone().to(self.device).squeeze(1) - torch.sum(cube_lr_c[:,channnel_list,:,:],dim=1),(w_local_c[:,i,:,:] + 1e-23)) 
            restore_hr[:,i,:,:] = torch.div(P.clone().to(self.device).squeeze(1) - torch.sum(cube_hr_c[:,channnel_list,:,:],dim=1),(w_ext_c[:,i,:,:] + 1e-23)) 

        loss_restore_lr = self.L1Loss(torch.clip(restore_lr,0,1).to(self.device),M.to(self.device))
        loss_restore_hr = self.L1Loss(torch.clip(restore_hr,0,1).to(self.device),X.to(self.device))
        loss_restore = 0.5 * loss_restore_hr  + 0.5 * loss_restore_lr 

        if torch.isinf(loss_restore).any() or torch.isnan(loss_restore).any():
            loss = loss_content
        else:
            loss = 0.5 * loss_content + 0.5 * loss_restore * 2

        return loss

    def img2tf(self,image_np):
        """
            When we process, the numpy data is normalized in range of 0 ~ 1
            Thus we turn the type ndarray to tensor 
        """
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


def predict_coefficients(X_np,P_np,M_np,L_np,device):
    """  Model Generate """
    model = network()
    trainer = Trainer(model,device) 
    """  Pre-process data """
    X = trainer.img2tf(X_np)
    P = trainer.img2tf(P_np)
    M = trainer.img2tf(M_np)
    L = trainer.img2tf(L_np)
    P_gt = np.clip(np.round(P_np * 255),0,255).astype(np.uint8)
    L_gt = np.clip(np.round(L_np * 255),0,255).astype(np.uint8)        
    trainer.to(device)
    trainer.train()
    if os.path.exists('./model/rnet.pth'):
        n_epochs,psnr_minimum = 100,0
    else:
        os.makedirs('./model')
        n_epochs,psnr_minimum = 1000,0
    n_epochs_bar = tqdm(range(n_epochs))
    for epoch in n_epochs_bar:
        # train
        trainer.train()
        trainer.optimizer.zero_grad()
        loss = trainer(X,P,M,L)
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        # Eval
        trainer.eval()
        with torch.no_grad():
            w_local,w_ext = trainer.model(X.to(device))
            lr = torch.sum(w_local.to(device)*M.to(device),dim=1).unsqueeze(1)
            hr = torch.sum(w_ext.to(device)*X.to(device),dim=1).unsqueeze(1)
            hr_pseudo = np.clip(np.round(trainer.tf2img(hr).astype(np.float32) * 255),0,255).astype(np.uint8)
            lr_pseudo = np.clip(np.round(trainer.tf2img(lr).astype(np.float32) * 255),0,255).astype(np.uint8)
            pass
        # Writer
        psnr_hr = PSNR(hr_pseudo,P_gt)
        psnr_lr = PSNR(lr_pseudo,L_gt)

        if psnr_minimum < psnr_hr :
            psnr_minimum = psnr_hr
            torch.save(trainer.model.state_dict(),'./model/rnet.pth')
            pass
        n_epochs_bar.set_description('Epoch : {}/{} PSNR(HR): {:.3f} PSNR(LR): {:.3f} Loss : {:.6f}'.format(epoch,n_epochs,psnr_hr,psnr_lr,loss.item()))
        pass
    # Output
    trainer.model.load_state_dict(torch.load('./model/rnet.pth'))
    trainer.eval()
    w_local,w_ext = trainer.model(X.to(device))
    coefficients_hr_np = w_ext.detach().squeeze(0).permute(1,2,0).cpu().numpy()
    return coefficients_hr_np

if __name__ == '__main__':
    print('Hello World')