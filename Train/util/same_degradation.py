import cv2
import os
import numpy as np
import random
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from util.unprocess import unprocess, random_noise_levels, add_noise
from util.process import process
# from unprocess import unprocess, random_noise_levels, add_noise
# from process import process
# import utils_image as util

from PIL import Image

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
def get_degrade_seq(img_shape):
    degrade_seq = []
    need_shift = False
    global_sf = None

    # -----------------------
    # isotropic gaussian blur
    # -----------------------
    B_iso = {
        "mode": "blur",
        "kernel_size": random.choice([5, 7, 9, 11, 13, 15, 17]),
        "is_aniso": False,
        "sigma": random.uniform(0.1, 2.8),
    }
    

    # -------------------------
    # anisotropic gaussian blur
    # -------------------------
    B_aniso = {
        "mode": "blur",
        "kernel_size": random.choice([5, 7, 9, 11, 13, 15, 17]),
        "is_aniso": True,
        "x_sigma": random.uniform(0.5, 8),
        "y_sigma": random.uniform(0.5, 8),
        "rotation": random.uniform(0, 180)
    }
    mode = random.random()
    if mode > 0.75:
        degrade_seq.append(B_iso)
    elif mode > 0.40:
        degrade_seq.append(B_aniso)
    elif mode > 0.2:
        degrade_seq.append(B_iso)
        degrade_seq.append(B_aniso)


    # -----------
    # down sample
    # -----------
    B_down = {
        "mode": "down",
        "sf": random.uniform(2, 6)
    }
    mode = random.random()
    if mode > 0.9:
        B_down["down_mode"] = "nearest"
        B_down["sf"] = random.choice([2, 4, 6])
        need_shift = True
    elif mode > 0.5:
        B_down["down_mode"] = "bilinear"
    else:
        B_down["down_mode"] = "bicubic"
    # elif:
    #     down_mode = random.choice(["bilinear", "bicubic"])
    #     up_mode = random.choice(["bilinear", "bicubic"])
    #     up_sf = random.uniform(0.5, B_down["sf"])
    #     B_down["down_mode"] = down_mode
    #     B_down["sf"] = B_down["sf"] / up_sf
    #     B_up = {
    #         "mode": "down",
    #         "sf": up_sf,
    #         "down_mode": up_mode
    #     }
    #     degrade_seq.append(B_up)

    degrade_seq.append(B_down)
    global_sf = B_down["sf"]

    # --------------
    # gaussian noise
    # --------------
    B_noise = {
        "mode": "noise",
        "noise_level": random.randint(1, 19),
        "noise": np.random.normal(0, random.randint(1, 19), img_shape),
    }
    if random.random() > 0.3:
        degrade_seq.append(B_noise)

    # ----------
    # jpeg noise
    # ----------
    B_jpeg = {
        "mode": "jpeg",
        "qf": random.randint(30, 85)
    }
    if random.random() > 0.3:
        degrade_seq.append(B_jpeg)

    # -------------------
    # Processed camera sensor noise
    # -------------------
    B_camera = {
        "mode": "camera",
    }
    # if random.random() > 0.75:
    #     degrade_seq.append(B_camera)

    # -------
    # shuffle
    # -------
    random.shuffle(degrade_seq)

    # ---------------
    # last jpeg noise
    # ---------------
    B_jpeg_last = {
        "mode": "jpeg",
        "qf": random.randint(30, 85)
    }
    if random.random() > 0.5:
        degrade_seq.append(B_jpeg_last)

    # --------------------
    # restore correct size
    # --------------------
    B_restore = {
        "mode": "restore",
        "sf": global_sf,
        "need_shift": need_shift,
        "up_mode": random.choice(["bilinear", "bicubic"])
    }

    degrade_seq.append(B_restore)
    return degrade_seq

def degrade_process(img, h, w, degrade_seq):
    for degrade_dict in degrade_seq:
        mode = degrade_dict["mode"]
        if mode == "blur":
            img = get_blur(img, degrade_dict)
        elif mode == "down":
            img = get_down(img, degrade_dict)
        elif mode == "noise":
            img = get_noise_without_level(img, degrade_dict)
        elif mode == 'jpeg':
            img = get_jpeg(img, degrade_dict)
        elif mode == 'camera':
            img = get_camera(img, degrade_dict)
        elif mode == 'restore':
            img = get_restore(img, h, w, degrade_dict)
        else:
            exit(mode)
    return img

def degradation_pipeline(img, img2=None):
    h, w, c = np.array(img).shape
    degrade_seq = get_degrade_seq((h, w, c))
    img = degrade_process(img, h, w, degrade_seq)
    if img2 is not None:
        img2 = degrade_process(img2, h, w, degrade_seq)
        # print_degrade_seg(degrade_seq)
    return img, img2


def get_blur(img, degrade_dict):

    img = np.array(img)
    k_size = degrade_dict["kernel_size"]
    if degrade_dict["is_aniso"]:
        sigma_x = degrade_dict["x_sigma"]
        sigma_y = degrade_dict["y_sigma"]
        angle = degrade_dict["rotation"]
    else:
        sigma_x = degrade_dict["sigma"]
        sigma_y = degrade_dict["sigma"]
        angle = 0

    kernel = np.zeros((k_size, k_size))
    d = k_size // 2
    for x in range(-d, d+1):
        for y in range(-d, d+1):
            kernel[x+d][y+d] = get_kernel_pixel(x, y, sigma_x, sigma_y)
    M = cv2.getRotationMatrix2D((k_size//2, k_size//2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
    kernel = kernel / np.sum(kernel)
    img = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='reflect')

    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_down(img, degrade_dict):

    img = np.array(img)
    sf = degrade_dict["sf"]
    mode = degrade_dict["down_mode"]
    h, w, c = img.shape
    if mode == "nearest":
        img = img[0::sf, 0::sf, :]
    elif mode == "bilinear":
        new_h, new_w = int(h/sf), int(w/sf)
        img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
    elif mode == "bicubic":
        new_h, new_w = int(h/sf), int(w/sf)
        img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_noise(img, degrade_dict):
    noise_level = degrade_dict["noise_level"]
    img = np.array(img)
    img = img + np.random.normal(0, noise_level, img.shape)
    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))

def get_noise_without_level(img, degrade_dict):
    noise = degrade_dict["noise"]
    img = np.array(img)
    h, w, c = img.shape
    img = img + noise[:h, :w, :c]
    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_jpeg(img, degrade_dict):
    qf = degrade_dict["qf"]
    img = np.array(img)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),qf] # (0,100),higher is better,default is 95
    _, encA = cv2.imencode('.jpg',img,encode_param)
    Img = cv2.imdecode(encA,1)
    return Image.fromarray(np.uint8(np.clip(Img, 0.0, 255.0)))


def get_camera(img, degrade_dict):
    h, w = img.size
    if h // 2 == 0 and w // 2 == 0:
        img = torch.from_numpy(np.array(img)) / 255.0
        deg_img, features = unprocess(img)
        shot_noise, read_noise = random_noise_levels()
        deg_img = add_noise(deg_img, shot_noise, read_noise)
        deg_img = deg_img.unsqueeze(0)
        features['red_gain'] = features['red_gain'].unsqueeze(0)
        features['blue_gain'] = features['blue_gain'].unsqueeze(0) 
        features['cam2rgb'] = features['cam2rgb'].unsqueeze(0) 
        deg_img = process(deg_img, features['red_gain'], features['blue_gain'], features['cam2rgb'])
        deg_img = deg_img.squeeze(0)
        deg_img = torch.clamp(deg_img * 255.0, 0.0, 255.0).numpy()
        deg_img = deg_img.astype(np.uint8)
        return Image.fromarray(deg_img)
    else:
        return img

def get_restore(img, h, w, degrade_dict):
    need_shift = degrade_dict["need_shift"]
    sf = degrade_dict["sf"]
    img = np.array(img)
    mode = degrade_dict["up_mode"]
    if mode == "bilinear":
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
    if need_shift:
        img = shift_pixel(img, int(sf))
    return Image.fromarray(img)

def get_kernel_pixel(x, y, sigma_x, sigma_y):
    return 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-((x*x/(2*sigma_x*sigma_x))+(y*y/(2*sigma_y*sigma_y))))


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def print_degrade_seg(degrade_seq):
    for degrade_dict in degrade_seq:
        print(degrade_dict)


if __name__ == "__main__":
    FacePath = '/data/vdb/lxmF/RealD/TrainData/FFHQ512PNG'
    SaveFacePath = './FaceSameDeg'
    SaveFacePathHQ = './FaceSameHQ'
    NaturalPath = '/data/vdb/lxmF/SRTrainData/DF2K_HighPatches256'
    SaveNaturalPath = 'NaturalSameDeg'
    SaveNaturalPathHQ = 'NaturalSameHQ'
    sf = 4
    os.makedirs(SaveFacePath, exist_ok=True)
    os.makedirs(SaveFacePathHQ, exist_ok=True)
    os.makedirs(SaveNaturalPath, exist_ok=True)
    os.makedirs(SaveNaturalPathHQ, exist_ok=True)
    # if not os.path.exists(SaveFacePath):
    #     os.makedirs(SaveFacePath)
    # if not os.path.exists(SaveNaturalPath):
    #     os.makedirs(SaveNaturalPath)
    
    ImgPaths = make_dataset(FacePath)
    NaturalPaths = make_dataset(NaturalPath)
    random.shuffle(NaturalPaths)
    # print(NaturalPaths)
    # random.shuffle(ImgPaths)
    for i, FacePath in enumerate(ImgPaths):
        FaceName = os.path.split(FacePath)[-1]
        NaturalPath = NaturalPaths[i]
        print([i, FaceName])
        # Img = Image.open(ImgPath).convert('RGB')
        # Img = Img.resize((256,256), Image.BICUBIC)
        FaceImg = Image.open(FacePath).convert('RGB')
        NaturalImg = Image.open(NaturalPath).convert('RGB')
        FaceHQ = FaceImg.resize((256,256), Image.BICUBIC)
        NaturalHQ = NaturalImg.resize((256,256), Image.BICUBIC)
        
        FaceLQ, NaturalLQ = degradation_pipeline(FaceHQ, NaturalHQ)
        
        FaceLQ =  FaceLQ.resize((256, 256), Image.BICUBIC)
        NaturalLQ =  NaturalLQ.resize((256, 256), Image.BICUBIC)
        FaceLQ.save(os.path.join(SaveFacePath, FaceName))
        FaceHQ.save(os.path.join(SaveFacePathHQ, FaceName))
        NaturalLQ.save(os.path.join(SaveNaturalPath, FaceName))
        NaturalHQ.save(os.path.join(SaveNaturalPathHQ, FaceName))
        # exit('rr')
        if i > 100:
            break
    # import glob
    # import os
    # import sys 

    # test_blur = {
    #     "mode": "blur",
    #     "kernel_size": 21,  # random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
    #     "is_aniso": False,
    #     "x_sigma": 8,  # random.uniform(0.5, 8),
    #     "y_sigma": 8,  # random.uniform(0.5, 8),
    #     "sigma": 2.8, # sigma
    #     "rotation": random.uniform(0, 180)
    # }
    # test_down = {
    #     "mode": "down",
    #     "sf": 2,
    #     "down_mode": "nearest"
    # }
    # test_noise = {
    #     "mode": "noise",
    #     "noise_level": 23
    # }
    # test_jpeg = {
    #     "mode": "jpeg",
    #     "qf": 10
    # }
    # test_camera = {
    #     "mode": "camera",
    # }
    # test_restore = {
    #     "mode": "restore",
    #     "sf": 2,
    #     "need_shift": False
    # }
    # img = cv2.imread("./00018.png")
    # # img = Image.open(sys.argv[1])
    # print(np.array(img).shape)
    # # h, w, c = img.shape
    # # blur_img = get_blur(img, test_blur)
    # # down_img = get_down(img, test_down)
    # # noise_img = get_noise(img, test_noise)
    # # jpeg_img = get_jpeg(img, test_jpeg)
    # # camera_img = get_camera(img, test_camera)
    # # restore_img = get_restore(down_img, h, w, test_restore)
    # for i in range(100):
    #     print(i)
    #     restore_img = degradation_pipeline(img)
    #     cv2.imwrite("./Tmp/deg_{:04d}.png".format(i), np.array(restore_img))
    # # cv2.imwrite('./camera.png', np.array(camera_img))
    # # cv2.imwrite("./deg.png", camera_img)

    # # img_list = glob.glob('./tiny_test/*.png')
    # # save_dir = "./tiny_test" + '_prac'
    # # if not os.path.exists(save_dir):
    # #     os.makedirs(save_dir)
    # # for img_path in img_list:
    # #     img = cv2.imread(img_path)
    # #     img_name = os.path.basename(img_path)
    # #     # print(img_name)
    # #     img = degradation_pipeline(img)
    # #     cv2.imwrite(os.path.join(save_dir, img_name), img)
