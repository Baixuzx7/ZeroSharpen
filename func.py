import cv2
import numpy as np
import scipy.fft
import imageio.v2 as imageio
from math import ceil, floor
from tnet import predict_spatial_info
from rnet import predict_coefficients
import torch

def VarPS(MS,PP,D,L):
    [h,w,c] = MS.shape   
    [H,W] = PP.shape
    v1,v2,v3 = 1,0.2,0.0001
    eps = 1e-28
    ratio = 4
    iterations = 50
    Lipschitz = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fx = np.asarray([1, -1]).reshape([1, 2])
    fy = np.asarray([1, -1]).reshape([2, 1])
    yp0,yp90 = gradient(PP,1),gradient(PP,3)
    [mm,nn] = PP.shape
    otf_fx = psf2otf(fx,[mm,nn])
    otf_fy = psf2otf(fy,[mm,nn])
    otf_abs = np.power(np.abs(otf_fx),2) + np.power(np.abs(otf_fy),2)
    Y_out = np.zeros([H,W,c])
    x = resample(MS,ratio)
    y = x
    tnew = 1
    for epoch in range(iterations):
        told = tnew
        xp = x
        dd = resample(x, 1/ratio )  - MS
        df = resample(dd, ratio)
        yg = y - df/Lipschitz  
        band_coefficients_group = predict_coefficients(x.astype(np.float32),PP.astype(np.float32),MS.astype(np.float32),L.astype(np.float32),device)
        x_folder = np.zeros_like(x)
        for band in range(c):
            xt,M = x[:,:,band],MS[:,:,band]
            xg0 = predict_spatial_info(X_np = xt,
                        P_np = PP,
                        D_np = D[:,:,band],
                        L_np = L,
                        Q_hr_np = guidedfilter(yp0, gradient(xt,1), 2, eps),
                        Q_lr_np = guidedfilter(gradient(L,1),gradient(M,1),r=2,eps=1e-28),
                        M_gt_np = gradient(M,1), 
                        device = device,
                        recoder = 'dx',
                        band_id = band)
            xg90 = predict_spatial_info(X_np = xt,
                        P_np = PP,
                        D_np = D[:,:,band],
                        L_np = L,
                        Q_hr_np = guidedfilter(yp90, gradient(xt,3) , 2, eps),
                        Q_lr_np = guidedfilter(gradient(L,3),gradient(M,3),r=2,eps=1e-28),
                        M_gt_np = gradient(M,3), 
                        device = device,
                        recoder = 'dy',
                        band_id = band)
            xr = PP
            for j in range(c):
                if j != band:
                    xr = xr - band_coefficients_group[:,:,j] * x[:,:,j]
            xr = xr / band_coefficients_group[:,:,band]
            xr = np.clip(xr,0,1)    
            weight = v1 * Lipschitz + v2 * otf_abs + v3 
            fftyg = scipy.fft.fft2(yg[:,:,band])
            fftx0 = np.conj(otf_fx) * scipy.fft.fft2(xg0)
            fftx90 = np.conj(otf_fy) * scipy.fft.fft2(xg90)
            fftxr = scipy.fft.fft2(xr)
            S = (v1 * Lipschitz * fftyg + v2 * (fftx0 + fftx90)  + v3 * fftxr ) / weight
            xf = scipy.fft.ifft2(S)
            xs = np.abs(xf)
            x_folder[:,:,band] = np.clip(xs,0,1)
            pass
        x = np.clip(x_folder,0,1)
        tnew = (1 + np.sqrt(1 + 4 * told * told)) / 2
        y = x + ((told - 1) / tnew) * (x - xp)
        pass
    Y_out = x
    return Y_out

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

    return imDst

def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf

def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)
    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP
    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI
    a = covIp / (varI + eps)
    b = meanP - a * meanI
    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N
    q = meanA * I + meanB
    return q

def gradient(x,flag):
    x = x.astype(np.float32)
    if flag == 1:
        y = np.concatenate([x, x], axis=1)[:,1:1 + x.shape[1]]
        return y - x
    elif flag == 3:
        y = np.concatenate([x, x], axis=0)[1:1 + x.shape[1],:]
        return y - x

def shrinkage(x,A):
    [m,n] = x.shape
    z = np.zeros([m,n])
    s = np.sign(x) * np.maximum(np.abs(x-A),z)
    return s

def resample(image,ratio,method = cv2.INTER_CUBIC):
    image_out = imresize(image,ratio)
    return image_out

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

def Gram_Schmidt_coefficient(imageHR,imageP):
    HR = imageHR.astype(np.float32)
    P  = imageP.astype(np.float32)
    h,w,c = HR.shape
    band_coefficients = np.ones([c,1])
    A = np.zeros([c,c])
    B = np.zeros([c,1])
    for i in range(c):
        for j in range(c):
            A[i,j] = np.sum(np.sum(HR[:,:,i] * HR[:,:,j]))
            pass
        B[i] = np.sum(np.sum(P * HR[:,:,i]))
        pass
    tau = 5
    iter = 150000
    gamma1 = 1/200000
    inv = np.linalg.inv((np.eye(c) + 2*tau*gamma1*A))
    for i in range(iter):
        band_coefficients = inv.dot(band_coefficients + 2*tau*np.maximum(-band_coefficients,0)+2*tau*gamma1*B)
        pass 
    return band_coefficients
    
def multi_band_imread(path,idx,h,w,bands = 8):
    image_container = np.zeros([h,w,bands])
    for i in range(1,1+8):
        image_path = path + str(i) + '/' + str(idx) + '.tif'
        image = imageio.imread(image_path)
        image_container[:,:,i-1] = image
        pass
    return image_container.astype(np.uint8)

def multi_band_imwrite(image,path,idx,bands = 8):
    for i in range(bands):
        image_b = image[:,:,i]
        save_path = path + str(i) + '/' + str(idx) + '.tif'
        imageio.imwrite(save_path,image_b)

if __name__ == '__main__':
    print('Hello World')