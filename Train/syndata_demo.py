import cv2
import numpy as np
from alphabets import alphabet
import yaml
from collections import OrderedDict
from PIL import Image, ImageFilter, ImageDraw, ImageFont

import torch
import torch.utils.data as data
import random
from util.real_esrgan_bsrgan_degradation import real_esrgan_degradation, bsrgan_degradation
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
import time
import re
import os

import imgaug.augmenters as ia
import sys
import os.path as osp
import math
from torch.nn import functional as F


class TextDegradationDataset(data.Dataset):
    def __init__(self, opt):
        super(TextDegradationDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.dataroot = opt['path_bg']
        self.gt_folder = opt['path_bg']
        self.mean = opt['mean']
        self.std = opt['std']
        self.max_corpus_length = opt['max_text_length']
        self.CheckNum = int(opt['check_num'])
        self.CommonWords = alphabet

        ## label for A~Z a~z and 0~9
        #A~Za~z
        AZ_az = [3074, 3914, 1959, 165, 6064, 3762, 5455, 5370, 4753, 6449, 2607, 4368, 6344, 1064, 2616, 1024, 1958, 841, 3278, 5870, 3177, 4449, 5888, 1637, 333, 2059,5129, 2559, 302, 5076, 434, 5670, 2217, 6021, 6445, 2913, 5243, 3790, 2037, 665, 4333, 2034, 2404, 3906, 3671, 5036, 4053, 2679, 3486, 6071, 114, 3230]
        # 0~9
        number_letter = [575, 2116, 1230, 1857, 3157, 1564, 4124, 3708, 2072, 355]
        self.EnglishOnly = []
        self.NumberOnly = []
        for ind in AZ_az:
            self.EnglishOnly.append(alphabet[ind])
        for ind in number_letter:
            self.NumberOnly.append(alphabet[ind])


        self.corpus1, self.corpus2, self.corpus3 = None, None, None
        if opt['corpus_path1']:
            with open(opt['corpus_path1'], 'r') as f:
                lines = f.read().split('\n')
            self.corpus1 = [line.strip() for line in lines if len(line)>opt['min_text_length']]
            print("[Corpus1] Number of Corpus1 pairs: {}, classes: {}".format(len(self.corpus1), len(self.CommonWords)))

        if opt['corpus_path2']:
            with open(opt['corpus_path2'], 'r') as f:
                lines = f.read().split('\n')
            self.corpus2 = [line.strip() for line in lines if len(line)>opt['min_text_length']]
            print("[Corpus2] Number of Corpus2 pairs: {}, classes: {}".format(len(self.corpus2), len(self.CommonWords)))

        if opt['corpus_path3']:
            with open(opt['corpus_path3'], 'r') as f:
                lines = f.read().split('\n')
            self.corpus3 = [line.strip() for line in lines if len(line)>opt['min_text_length']]
            print("[Corpus3] Number of Corpus3 pairs: {}, classes: {}".format(len(self.corpus3), len(self.CommonWords)))
        
        if self.corpus2 == None:
            self.corpus2 = self.corpus1
        if self.corpus3 == None:
            self.corpus3 = self.corpus1

        self.gray_aug = ia.Grayscale(alpha=random.randint(5,10)/10.0)

        if self.io_backend_opt['type'] == 'disk':
            ##background sample
            self.paths = []
            bgs = os.listdir(self.dataroot)
            for bgi in bgs:
                self.paths.append(osp.join(self.dataroot, bgi))

            index = np.arange(len(self.paths))
            np.random.shuffle(index)
            self.paths = np.array(self.paths)
            self.paths = self.paths[index]
            print("[Dataset] Number of Background pairs:", len(self.paths))


            ##font sample
            self.font_paths = []
            Fonts = os.listdir(opt['path_font'])
            for f in Fonts:
                self.font_paths.append(osp.join(opt['path_font'], f))
            print("[FontType] Number of Font styles: {} ".format(len(self.font_paths)))

        else: 
            # disk backend: scan file list from a folder
            raise ValueError("Error in io_backend_opt. Only support disk")


        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
    
    def make_dataset(self, dirs):
        images = []
        assert os.path.isdir(dirs), '%s is not a valid directory' % dirs
        for root, _, fnames in sorted(os.walk(dirs)):
            fnames.sort()
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images
    
    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def AddDownSample(self,img): # downsampling
        sampler = random.randint(15, 35)*1.0
        h0, w0 = img.shape[:2]
        if random.random() > 0.5:
            img = cv2.resize(img, (int(w0/sampler*10.0), int(h0/sampler*10.0)), cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (int(w0/sampler*10.0), int(h0/sampler*10.0)), cv2.INTER_LINEAR)
        out = cv2.resize(img.copy(), (self.out_size//2, self.out_size//2), cv2.INTER_LINEAR)
        return out


    
    def image_add_text(self, img):
        '''
        Get text and label
        '''
        text, label = self.get_text()
        while self.check_corpus(text): # double check selected text and label
            text, label = self.get_text()     

        w, h = img.size
        if random.random() > 0.92: # 0.96: #white bg
            img = Image.new('RGB', (w, h), (random.randint(0,255),random.randint(0,255),random.randint(0,255)))

        fontpath = self.font_paths[random.randint(0, len(self.font_paths)-1)]

        text_size = random.randint(90,140)#
        x = random.randint(-10, 20) #
        y = random.randint(-20, 10) #
        pos = (x, y) #width, height from top left to bottom right
        fontStyle = ImageFont.truetype(fontpath, text_size, encoding="utf-8")

        pos_mask = Image.new('L', (w, h), 0)
        drawer_tmp = ImageDraw.Draw(pos_mask)
        char_locs = []
        img_max_width = 0
        text_add_space = text
        for i in range(1, len(text_add_space)+1):
            if text_add_space[i-1] == ' ':
                continue
            p = text_add_space[:i]
            drawer_tmp.text(pos, p, font=fontStyle, fill=255)
            char_mask = np.array(pos_mask).copy()
            vertical_projection = np.sum(char_mask, axis=0)
            ws = np.where(vertical_projection>1) #height, width
            locs = list(ws[0])
            if len(locs) == 0:
                continue
            if len(char_locs) == 0:
                char_locs.append(max(min(locs), 0))
                char_locs.append(min(max(locs), w-1))
            else:
                new_locs = []
                for l in locs:
                    if l > char_locs[-1] and l - char_locs[-1] > 2:
                        new_locs.append(l)
                if len(new_locs):
                    char_locs.append(max(min(new_locs), 0))
                    char_locs.append(min(max(new_locs), w-1))
        

        if len(char_locs) == 0:
            print(['error max char_locs', text])
            return None, np.zeros((1,1)), None, None, None

        img_max_width = max(char_locs)
        if len(text)  != len(char_locs) // 2 or len(label) != len(char_locs) // 2 or img_max_width > 128 * self.CheckNum:
            return None, np.zeros((1,1)), None, None, None
        
        if self.CheckNum == 1:
            char_locs = [0,  128]

        
        for i in range(len(text), self.CheckNum):
            char_locs.append(self.CheckNum*128)
            char_locs.append(self.CheckNum*128)

        if random.random() > 0.9:
            text_color = (0,0,0)
        else:
            text_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        
        drawer = ImageDraw.Draw(img)
        drawer.text(pos, text_add_space, font=fontStyle, fill=text_color)
        pos_mask = np.array(pos_mask)
        mask = np.repeat(pos_mask[:,:, np.newaxis], 3, axis=2)
        mask[mask>128]=255
        mask[mask<=128]=0
        
        img = np.array(img)[:,:,::-1].astype(np.float32)
        offset_w = min(img_max_width + random.randint(0,16), 128*self.CheckNum)
        offset_w = offset_w // 4 * 4
        
        img = img[:, :offset_w, :]
        mask = mask[:, :offset_w, :]
        
        if img.shape[-2] < 10 or img.shape[-3] < 10 or img.shape[-2] > 128 * self.CheckNum:
            return None, np.zeros((1,1)), None, None, None
        return img / 255.0, mask /255.0, text, label, char_locs

    def read_image(self, gt_path):
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path)
                img_gt = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                gt_path = self.paths[random.randint(0, len(self.paths)-1)]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        return img_gt

    def get_background_patch(self, gt_path):
        img_gt = self.read_image(gt_path) # 400*400
        ##flip
        flip = 1 if random.random() > 0.5 else 0
        if flip:
            cv2.flip(img_gt, 1, img_gt)
        random_size = random.randint(320,400)
        img_gt = cv2.resize(img_gt, (random_size,random_size), cv2.INTER_LINEAR)
        h0, w0 = img_gt.shape[:2]

        h1 = np.random.choice(np.arange(0, h0//2))
        w1 = np.random.choice(np.arange(0, w0//4))
    
        crop_size = min(random.randint(w0//4,w0//4*3), 128) # < 400 ori

        img_gt = img_gt[h1:h1+crop_size//self.CheckNum, w1:w1+crop_size]
        img_gt = cv2.resize(img_gt, (128*self.CheckNum,128), cv2.INTER_LINEAR)
        return img_gt

    

    def check_corpus(self, text):
        for i in text:
            if i not in self.CommonWords:
                return True
        if len(text)> self.CheckNum:
            return True
        return False
    
    def get_text(self):
        which_text = random.random()
        if which_text > 0.5: #Chinese Corpus
            which_corpus = random.random()
            if which_corpus > 0.7:
                text = self.corpus1[random.randint(0, len(self.corpus1) - 1)]
            elif which_corpus > 0.4:
                text = self.corpus2[random.randint(0, len(self.corpus2) - 1)]
            else:
                text = self.corpus3[random.randint(0, len(self.corpus3) - 1)]

        elif which_text > 0.2: #combination of characters from dictionary
            text = random.choices(self.CommonWords, k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
            random.shuffle(text)
            text = "".join(text)
            text = text[:int(self.opt['max_text_length'])]
            text = text.replace(' ', '').replace('\u3000', '') #remove space

        else:#english + number 
            symble = random.choices(self.EnglishOnly + self.NumberOnly, k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
            text = "".join(symble)
            text = text[:int(self.opt['max_text_length'])]
            text = text.replace(' ', '').replace('\u3000', '')

        ##remove bad characters
        text = "".join(text.split())
        text = text.encode('unicode_escape').decode('utf-8').replace(' ', '')
        result = re.findall(r'\\x[a-f0-9]{2}', text)
        for x in result:
            text = text.replace(x, '')
        try:
            text = text.encode('utf-8').decode('unicode_escape')
        except:
            return [], []
        len_text = len(text)

        if len_text > self.CheckNum:
            x = len_text - self.CheckNum
            x0 = random.randint(0, x)
            y0 = x0 + random.randint(self.opt['min_text_length'], self.max_corpus_length)
            select_text = text[x0:int(min(y0, self.max_corpus_length + x0))]
        else:
            select_text = text
        
        if self.CheckNum == 1:
            ind = random.randint(0, len(self.CommonWords)-1)
            select_text = self.CommonWords[ind]
            if len(select_text.replace('\u3000', '')) == 0:
                select_text = self.CommonWords[ind + random.randint(-100,100)]

        check_text = ''
        label = []
        for i in select_text:
            index = self.CommonWords.find(i)
            if index >= 0:
                check_text = check_text + i
                label.append(index)

        return check_text, label

    def __getitem__(self, index):
        index = random.randint(0, len(self.paths)-1)
        gt_path = self.paths[index]
        img_gt = self.get_background_patch(gt_path)

        im_PIL = img_gt[:,:,::-1]*255 # to RGB
        im_PIL = Image.fromarray(im_PIL.astype(np.uint8)) # RGB 0~255 H*W*C

        text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL) #output BGR 0~1
        while np.sum(mask_img) < 1.0: # remove these with null output
            text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL) #output BGR 0~1
        
        brightness = self.opt.get('brightness', (0.9, 1.1))
        contrast = self.opt.get('contrast', (0.9, 1.1))
        saturation = self.opt.get('saturation', (0.9, 1.1))
        # hue = self.opt.get('hue', (-0.1, 0.1))
        hue = self.opt.get('hue', None)
        text_img = self.color_jitter_pt(img2tensor(text_img, bgr2rgb=True, float32=False), brightness, contrast, saturation, hue)  #RGB Tensor 0~1 C*H*W
        text_img = text_img.numpy().transpose(1,2,0)[:,:,::-1] #transfer back to numpy for the following degradation, 0~1, BGR, H*W*C


        try:
            degradation_type = random.random()
            if degradation_type > 0.45:#real-esrgan
                ##input should be BGR 0~1 numpy H*W*C
                ##output is RGB 0~1 tensor 
                lq = real_esrgan_degradation(text_img, insf=random.choice([1,2,2,3,3,3])).squeeze(0).detach().numpy() #output numpy c*h*w 0~1 RGB
                lq = lq.transpose((1,2,0)) #transfer to h*w*c
            elif degradation_type > 0.01:#bsrgan
                ##input should be RGB 0~1 numpy H*W*C
                ##output is RGB 0~1 numpy H*W*C
                gt_tmp = text_img[:,:,::-1]#transfer to RGB
                lq, _ = bsrgan_degradation(gt_tmp, sf=random.choice([1,2,2,3,3,3]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
                lq = lq.astype(np.float32)
            else:
                lq = text_img[:,:,::-1] #out RGB
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(['error degradation', text_img.shape, e, exc_type, fname, exc_tb.tb_lineno])
            lq = np.ascontiguousarray(text_img[:,:,::-1]) #out RGB
        
        lq = np.clip(lq, 0, 1)

        h_hq, w_hq = text_img.shape[:2]

        lq = cv2.resize(lq, (int(32*w_hq/h_hq), 32),  interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))

        ##Norm the width by filling zeros
        TextGTFillBG = np.zeros((128, 128*self.CheckNum, 3)).astype(text_img.dtype)
        MaskFillBG = np.zeros((128, 128*self.CheckNum, 3))
        TextLQFillBG = np.zeros((32, 32*self.CheckNum, 3)).astype(lq.dtype)
        if text_img.shape[-2] < 128*self.CheckNum:
            TextGTFillBG[:, :text_img.shape[-2], :] = TextGTFillBG[:, :text_img.shape[-2], :] + text_img
            text_img = TextGTFillBG
            MaskFillBG[:, :mask_img.shape[-2], :] = MaskFillBG[:, :mask_img.shape[-2], :] + mask_img
            mask_img = MaskFillBG
        if lq.shape[-2] < 32*self.CheckNum:
            TextLQFillBG[:, :lq.shape[-2], :] = TextLQFillBG[:, :lq.shape[-2], :] + lq
            lq = TextLQFillBG
        if lq.shape[-2] > 32*self.CheckNum or text_img.shape[-2] > 128*self.CheckNum:
            lq = cv2.resize(lq, (32*self.CheckNum, 32), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
        
        if len(label_gt) < self.CheckNum:
            for _ in range(len(text), self.CheckNum):
                label_gt.append(6735)


        text_img = img2tensor(text_img, bgr2rgb=True, float32=False) #RGB 0~1
        mask_img = img2tensor(mask_img, bgr2rgb=True, float32=False) #RGB 0~1
        lq = img2tensor(lq, bgr2rgb=False, float32=False) #RGB 0~1


        # normalize
        normalize(text_img, self.mean, self.std, inplace=True) #-1~1 RGB
        normalize(lq, self.mean, self.std, inplace=True) #-1~1 RGB


        label = torch.Tensor(label_gt).type(torch.LongTensor)
        char_locs = torch.Tensor(char_locs) / (self.CheckNum*128) #
        return {'gt': text_img, 'mask':mask_img, 'label':label, 'lq': lq, 'boxinfo':char_locs} 

    def __len__(self):
        return max(len(self.paths), 1000)


def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

if __name__ == "__main__":
    from tqdm import tqdm
    with open('./options/train.yml', mode='r') as f:
        fullopt = yaml.load(f, Loader=ordered_yaml()[0])
    opt = fullopt['datasets']['train']
    save_path = './syn_data_samples'
    os.makedirs(save_path, exist_ok=True)

    s_name = time.strftime('%m-%d-%H_%M_%S', time.localtime())

    
    DataCls = TextDegradationDataset(opt)
    x = DataCls[0]

    ## 1 LQ 
    lq = x['lq']
    lqTmp = lq * 0.5 + 0.5
    lqTmp = lqTmp.permute(1, 2, 0).flip(2) # RGB->BGR
    lqTmp = np.clip(lqTmp.float().cpu().numpy(), 0, 1) * 255.0
    cv2.imwrite(osp.join(save_path, '{}_lq.png'.format(s_name)), lqTmp)

    ## 2 GT
    gt = x['gt']
    gtTmp = gt * 0.5 + 0.5
    gtTmp = gtTmp.permute(1, 2, 0).flip(2) # RGB->BGR
    gtTmp = np.clip(gtTmp.float().cpu().numpy(), 0, 1) * 255.0
    cv2.imwrite(osp.join(save_path, '{}_gt.png'.format(s_name)), gtTmp)

    ## 3 Prior
    mask = x['mask']
    maskTmp = mask.permute(1, 2, 0)
    maskTmp = np.clip(maskTmp.float().cpu().numpy(), 0, 1) * 255.0
    cv2.imwrite(osp.join(save_path, '{}_mask.png'.format(s_name)), maskTmp)

    ## 4 Locs 
    pad = 2
    padr = 2
    img_max_width = 2048
    ShowLocs = gt.clone()
    gt_locs = x['boxinfo'].unsqueeze(0) * img_max_width
    for b in range(gt_locs.size(0)):
        for l in range(0, gt_locs.size(1), 2):
            x, y = int(gt_locs[b][l].item()), int(gt_locs[b][l+1].item())
            ShowLocs[0, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowLocs[0, :64, max(0, x-pad):min(x+pad, img_max_width)]*0 + 1
            ShowLocs[0, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowLocs[0, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0 
            ShowLocs[1, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowLocs[1, :64, max(0, x-pad):min(x+pad, img_max_width)]*0 
            ShowLocs[1, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowLocs[1, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0
            ShowLocs[2, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowLocs[2, :64, max(0, x-pad):min(x+pad, img_max_width)]*0
            ShowLocs[2, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowLocs[2, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0 + 1
    
    ShowLocs = ShowLocs * 0.5 + 0.5
    ShowLocs = ShowLocs.permute(1, 2, 0).flip(2) # RGB->BGR
    ShowLocs = np.clip(ShowLocs.float().cpu().numpy(), 0, 1) * 255.0
    cv2.imwrite(osp.join(save_path, '{}_locs.png'.format(s_name)), ShowLocs)

    print('Results are saved in ' + save_path)

