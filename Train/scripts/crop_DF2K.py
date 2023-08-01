import os
import numpy as np
from PIL import Image
import random
import cv2
import os.path as osp        


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dirs):
    images = []
    assert os.path.isdir(dirs), '%s is not a valid directory' % dirs

    for root, _, fnames in sorted(os.walk(dirs)):
        fnames.sort()
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def GenerateSobelEdge(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize=3)
    y = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return edge, np.mean(edge), np.var(edge)
    
def GetHighPatch(Imgs, oS):
    w,h = Imgs.size
    w_offset_A = np.floor(random.randint(0, max(0, w - oS - 1))) # degradation shift
    h_offset_A = np.floor(random.randint(0, max(0, h - oS - 1))) #
    A = Imgs.crop((w_offset_A,h_offset_A,w_offset_A + oS,h_offset_A + oS))
    edge, m, v = GenerateSobelEdge(A)
    max_edge = edge
    max_m = m
    max_A = A
    max_v = v

    for_time = 3
    cur_num = 0
    while v < 1200:
        cur_num = cur_num + 1
        if cur_num >= for_time:
            break
        w_offset_A = np.floor(random.randint(0, max(0, w - oS - 1))) # degradation shift
        h_offset_A = np.floor(random.randint(0, max(0, h - oS - 1))) #
        A = Imgs.crop((w_offset_A,h_offset_A,w_offset_A + oS,h_offset_A + oS))
        edge, m, v = GenerateSobelEdge(A)
        if v > max_v:
            max_edge = edge
            max_A = A
            max_v = v
    if cur_num >= for_time:
        return max_edge, max_A, max_v, max_m
    else:
        return edge, A, v, m

if __name__ == '__main__':
    TestImgPath = '~/DIV2K_train_HR' # DF2K Path
    SavePath = '~/DF2K_Patch' # SavePath
    CropSize = 400
    ScaleLists = [6, 4, 2]
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    SplitNames = os.listdir(TestImgPath)
    SplitNames.sort()
    total = 0
    for split in SplitNames:
        fis = os.listdir(osp.join(TestImgPath, split))
        for fi in fis:
            KeyLists = os.listdir(osp.join(TestImgPath, split, fi))
            for key in KeyLists:
                ImgLists = os.listdir(osp.join(TestImgPath, split, fi, key))
                for ImgName in ImgLists:
                    ImgPath = osp.join(TestImgPath, split, fi, key, ImgName)
                    A = Image.open(ImgPath).convert('RGB')
                    w,h = A.size
                    print([split,fi,key,ImgName])
                    for ss in ScaleLists:
                        new_w = w*ss//10
                        new_h = h*ss//10
                        if ss < ScaleLists[0]:
                            A = A.resize((new_w, new_h), Image.LANCZOS)

                        HowmanyEach = min(new_w//CropSize, new_h//CropSize) * 2 * 2
                        for j in range(HowmanyEach):
                            _, Crop, V, M = GetHighPatch(A, CropSize)
                            total = total + 1
                            SaveName = '{}_{}_{}_{}_{:05d}_{:02d}.png'.format(split, fi, key, ImgName[:-4],j,ss)
                            Crop.save(os.path.join(SavePath, SaveName))

    