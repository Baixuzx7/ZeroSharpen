from utils import *
from func import *
from eval import *
import random
import os

""" Seed Setting"""
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    ms_file_name  = './data/ms/1.tif'
    pan_file_name = './data/pan/1.tif'
    image_P = imageio.imread(pan_file_name)
    image_GT = imageio.imread(ms_file_name)
    patch_MS = resample(image_GT,ratio=1/4,method = cv2.INTER_CUBIC)
    patch_PAN = resample(image_P,ratio=1/4,method = cv2.INTER_CUBIC)
    hrms = varPS(patch_MS,patch_PAN)
    if not os.path.exists('./result'):
         os.makedirs('./result')
    imageio.imwrite('./result/1.tif',hrms)
    pass