import numpy as np 
import scipy.signal
import scipy.io
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def ERGAS(I_ms,I_f):
    f,ms = I_f.astype(np.float32),I_ms.astype(np.float32)
    h,w,c = f.shape
    S = 0
    for i in range(c):
        A = np.mean(f[:,:,i])
        C = np.sqrt(np.sum(np.power(ms[:,:,i] - f[:,:,i],2))/h/w)
        S = S + (C/A)**2
    S_mean = S / c
    ergas = 25 * np.sqrt(S_mean)
    return ergas


def RMSE(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    if len(I_f.shape) == 2:
        h, w = f.shape
        c = 1
    else:
        h, w, c = f.shape
        pass
    D = np.power(ms - f,2)
    rmse = np.sqrt(np.sum(D)/h/w/c)
    return rmse


def RASE(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    C_sum = 0
    for i in range(c):
        C_sum = C_sum + np.sum(np.power(ms[:, :, i] - f[:, :, i], 2)) / h / w
    C_mean = C_sum / c
    rase = np.sqrt(C_mean) * 100 / np.mean(ms)
    return rase


def QAVE(I_ms,I_f):
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    ms_mean = np.mean(ms,axis=-1)
    f_mean = np.mean(f,axis=-1)
    Qx_sum,Qy_sum,Qxy_sum = 0,0,0
    for i in range(c):
        M = ms[:,:,i] - ms_mean
        Qx_sum = Qx_sum + np.power(M,2) 
        F = f[:, :, i] - f_mean
        Qy_sum = Qy_sum + np.power(F,2)
        Qxy_sum = Qxy_sum + M * F 
    Qx = (1/c - 1) * Qx_sum
    Qy = (1/c - 1) * Qy_sum
    Qxy = (1/c - 1) * Qxy_sum
    Q = (4 * Qxy * ms_mean * f_mean) / ( (Qx + Qy) * ( np.power(ms_mean,2) + np.power(f_mean,2) ) + 2.2204e-16)
    qave = np.sum( Q) / h / w
    return qave


def SSIM_Multi_Band(I_ms,I_f):
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    f, ms = I_f.astype(np.float32), I_ms.astype(np.float32)
    h, w, c = f.shape
    # Regularization Number
    ssim_score_sum = 0
    for i in range(c):
        ssim_score_sum = ssim_score_sum + ssim(f[:, :, i], ms[:, :, i]) 

    ssim_score = ssim_score_sum / c
    return ssim_score


def RSGenerate(image, percent, colorization=True):
    #   RSGenerate(image,percent,colorization)
    #                               --Use to correct the color
    # image should be R G B format with three channels
    # percent is the ratio when restore whose range is [0,100]
    # colorization is True
    m, n, c = image.shape
    # print(np.max(image))
    image_normalize = image / np.max(image)
    image_generate = np.zeros(list(image_normalize.shape))
    if colorization:
        # Multi-channel Image R,G,B
        for i in range(c):
            image_slice = image_normalize[:, :, i]
            pixelset = np.sort(image_slice.reshape([m * n]))
            maximum = pixelset[np.floor(m * n * (1 - percent / 100)).astype(np.int32)]
            minimum = pixelset[np.ceil(m * n * percent / 100).astype(np.int32)]
            image_generate[:, :, i] = (image_slice - minimum) / (maximum - minimum + 1e-9)
            pass
        image_generate[np.where(image_generate < 0)] = 0
        image_generate[np.where(image_generate > 1)] = 1
        image_generate = cv2.normalize(image_generate, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    return image_generate.astype(np.uint16)

class QualityIndex:
    def __init__(self,sensor):
        self.sensor = sensor
        self.filter = self.GetMTF_Filter()

    def GetMTF_Filter(self):
        if self.sensor == 'QB':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/QBfilter.mat')['QBfilter']
        elif self.sensor == 'IKONOS':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/IKONOSfilter.mat')['IKONOSfilter']
        elif self.sensor == 'GeoEye1':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/GeoEye1filter.mat')['GeoEye1filter']
        elif self.sensor == 'WV2':
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/WV2filter.mat')['WV2filter']
        else:
            MTF_Filter = scipy.io.loadmat('./MTF_PAN/nonefilter.mat')['nonefilter']
            pass
        return MTF_Filter

    def MTF_PAN(self,image_pan):
        pan = np.pad(image_pan,((20,20),(20,20)),mode='edge')
        image_pan_filter = scipy.signal.correlate2d(pan,self.filter,mode='valid')
        pan_filter = (image_pan_filter + 0.5).astype(np.uint8).astype(np.float32)
        return pan_filter

    def UQI(self,x,y):
        x = x.flatten()
        y = y.flatten()
        mx = np.mean(x)
        my = np.mean(y)
        C = np.cov(x, y)
        Q = 4 * C[0, 1] * mx * my / (C[0,0] + C[1, 1] + 1e-21) / (mx**2 + my**2 + 1e-21)
        return Q

    def D_s(self,fusion,ms,pan,S,q):
        D_s_index = 0
        h, w, c = fusion.shape
        pan_filt = self.MTF_PAN(pan)
        for i in range(c):
            band_fusion = fusion[:,:,i]
            band_pan = pan
            # 分块
            Qmap_high = []
            for y in range(0,h,S):
                for x in range(0,w,S):
                    Qmap_high.append(self.UQI(band_fusion[y:y+S,x:x+S],band_pan[y:y+S,x:x+S]))
                    pass
                pass
            Q_high = np.mean(np.asarray(Qmap_high))

            band_ms = ms[:, :, i]
            band_pan_filt = pan_filt
            # 分块
            Qmap_low = []
            for y in range(0, h, S):
                for x in range(0, w, S):
                    Qmap_low.append(self.UQI(band_ms[y:y + S, x:x + S], band_pan_filt[y:y+S,x:x+S]))
                    pass
                pass
            Q_low = np.mean(np.asarray(Qmap_low))
            D_s_index = D_s_index + np.abs(Q_high - Q_low)**q

        D_s_index = (D_s_index / c)**(1 / q)
        return D_s_index

    def D_lambda(self,fusion,ms,S,p):
        D_lambda_index = 0
        h, w, c = fusion.shape
        for i in range(0,c-1):
            for j in range(i+1,c):
                bandi = ms[:,:,i]
                bandj = ms[:,:,j]
                # 分块
                Qmap_exp = []
                for y in range(0, h, S):
                    for x in range(0, w, S):
                        Qmap_exp.append(self.UQI(bandi[y:y + S, x:x + S], bandj[y:y + S, x:x + S]))
                        pass
                    pass
                Q_exp = np.mean(np.asarray(Qmap_exp))

                bandi = fusion[:, :, i]
                bandj = fusion[:, :, j]
                # 分块
                Qmap_fused = []
                for y in range(0, h, S):
                    for x in range(0, w, S):
                        Qmap_fused.append(self.UQI(bandi[y:y + S, x:x + S], bandj[y:y + S, x:x + S]))
                        pass
                    pass
                Q_fused = np.mean(np.asarray(Qmap_fused))
                D_lambda_index = D_lambda_index + np.abs(Q_fused - Q_exp)**p

        s = (c**2 - c)/2
        D_lambda_index = (D_lambda_index/s)**(1/p)
        return D_lambda_index

    def QNR(self,fusion,ms,pan,S = 32,p = 1,q = 1,alpha = 1,beta = 1):
        # The size of the fusion, pan and ms is the same
        # ms (Use bicubic methoc to upsample)
        # The difference between the matlab and python comes from interpolation method
        h, w, c = fusion.shape
        ms_upsample = cv2.resize(ms,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
        D_lambda_index = self.D_lambda(fusion,ms_upsample,S,p)
        D_s_index = self.D_s(fusion,ms_upsample,pan,S,q)
        QNR_index = ((1 - D_lambda_index)**alpha) * ((1 - D_s_index)**beta)
        return D_lambda_index,D_s_index,QNR_index


if __name__ == '__main__':
    print('hello world')

    