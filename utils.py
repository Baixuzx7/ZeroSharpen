import numpy as np
from tqdm import tqdm
from func import VarPS as pansharpening
from func import resample

def varPS(ms,pan):
    # convert double
    imageL  = resample(pan,0.25).astype(np.float32)
    imageMS = ms.astype(np.float32)
    imageP  = pan.astype(np.float32)
    imageD  = resample(resample(ms,1/4),4).astype(np.float32)
    # normalization
    [h,w,c] = imageMS.shape
    [H,W] = imageP.shape
    band_maximum = np.zeros(c)
    imageMS_normal = np.zeros([h,w,c])
    imageD_normal = np.zeros_like(imageD)
    for i in range(c):
        band_maximum[i] = np.max(imageMS[:,:,i])
        imageMS_normal[:,:,i] = imageMS[:,:,i] / band_maximum[i]
        imageD_normal[:,:,i] = imageD[:,:,i] / np.max(imageD[:,:,i])
        pass
    imageP_normal = imageP / np.max(imageP)
    imageL_normal = imageL / np.max(imageL)
    # parameters
    imDst = pansharpening(imageMS_normal,imageP_normal,imageD_normal,imageL_normal)
    # restore image by multiplying the maximum value
    image_out = imDst * band_maximum
    return image_out.astype(np.uint8)

if __name__ == '__main__':
    print('Hello World')
 