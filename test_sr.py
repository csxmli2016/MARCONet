import torch
import cv2
import numpy as np
import os.path as osp
import time
from models import networks, ocr
import torchvision.transforms as transforms
from utils.alphabets import alphabet
import os
import argparse
import traceback
from utils.yolo_ocr_xloc import get_yolo_ocr_xloc
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def print_networks(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6

def get_labels_from_text(text):
    labels = []
    for t in text:
        index = alphabet.find(t)
        labels.append(index)
    return labels

def get_text_from_labels(TestPreds):
    PredsText = ''
    for i in range(len(TestPreds)):
        PredsText = PredsText + alphabet[TestPreds[i]]
    return PredsText



def main(L_path, save_path, manual_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelTSPGAN = networks.TSPGAN()
    modelTSPGAN.load_state_dict(torch.load('./checkpoints/net_prior_generation.pth')['params'], strict=True)
    modelTSPGAN.eval()

    modelSR = networks.TSPSRNet()
    modelSR.load_state_dict(torch.load('./checkpoints/net_sr.pth')['params'], strict=True)
    modelSR.eval()

    modelEncoder = networks.TextContextEncoderV2()
    modelEncoder.load_state_dict(torch.load('./checkpoints/net_transformer_encoder.pth')['params'], strict=True)
    modelEncoder.eval()


    yolo_character = YOLO('/mnt/sfs-common/xmli/Projects/1_Text/yolo/experiments/train2/weights/best.pt')
    real_ocr_pipeline = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')


    print('{:>28s} : {} M Parameters'.format('Transformer Encoder', print_networks(modelEncoder)))
    print('{:>28s} : {} M Parameters'.format('Structure Prior Network', print_networks(modelTSPGAN)))
    print('{:>28s} : {} M Parameters'.format('Super-Resolution Network', print_networks(modelSR)))
    
    
    print('#'*96)
    
    modelTSPGAN = modelTSPGAN.to(device)
    modelSR = modelSR.to(device)
    modelEncoder = modelEncoder.to(device)
    
    

    torch.cuda.empty_cache()
    
    
    img_names = os.listdir(L_path)
    img_names.sort()
    for img_name in img_names:
        img_path = osp.join(L_path, img_name)
        img_basename, ext = osp.splitext(osp.basename(img_path))

        '''
        Step 1: Predicting the character labels, bounding boxes.
        '''

        img, recognized_boxes, recognized_chars, char_x_centers = get_yolo_ocr_xloc(
            img_path,                        # Path to the input image file
            yolo_model=yolo_character,       # YOLO model instance for character detection
            ocr_pipeline=real_ocr_pipeline,  # OCR pipeline/model for character recognition
            num_cropped_boxes=5,             # Number of adjacent character boxes to include in each cropped segment (window size)
            expand_px=1,                     # Number of pixels to expand each crop region on all sides (except first/last)
            expand_px_for_first_last_cha=12, # Number of pixels to expand the crop region for the first and last character (left/right respectively)
            yolo_iou=0.1,                    # IOU threshold for YOLO non-max suppression (NMS)
            yolo_conf=0.07                   # Confidence threshold for YOLO detection
        )
        print(f"Image {img_path} recognized chars: {''.join(recognized_chars)}")


        h, w, c = img.shape
        ShowLQ = cv2.resize(img, (0,0), fx=128/h, fy=128/h,  interpolation=cv2.INTER_CUBIC)
        LQ = cv2.resize(img, (0,0), fx=32/h, fy=32/h,  interpolation=cv2.INTER_CUBIC)
        ori_lq_w = LQ.shape[1]


        TextLQFillBG = np.zeros((32, 32*16, 3)).astype(LQ.dtype)
        if LQ.shape[-2] <= 32*16:
            TextLQFillBG[:, :LQ.shape[-2], :] = TextLQFillBG[:, :LQ.shape[-2], :] + LQ
            LQ = TextLQFillBG
        else:
            print(['Warning!!! The width of the LQ text image exceeds the defined limit. Please crop it into shorter segments. (with height=32, width<=512)', LQ.shape])
            continue
            
        LQ = transforms.ToTensor()(LQ)
        LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)

        LQ = LQ.unsqueeze(0).to(device)


        # Compute preds_locs from recognized_boxes (YOLO output)
        # recognized_boxes: list of [x1, y1, x2, y2] in original image
        # LQ is resized to (32, LQ_width) before ToTensor
        num_boxes = len(recognized_boxes)
        preds_locs = torch.zeros(1, num_boxes * 2).float().to(device)
        # Get the width of the image after resizing to height 32
        lq_w = LQ.shape[-1]  # LQ is (1, 3, 32, lq_w) after ToTensor and unsqueeze
        # But we want the width before ToTensor, so use the numpy LQ before transforms
        lq_width = int(LQ.shape[-1])
        for i, box in enumerate(recognized_boxes):
            x1, y1, x2, y2 = box
            center = (x1 + x2) / 2.0
            width = (x2 - x1) / 2.0
            # Normalize to [0, 1] by dividing by original image width, then scale to lq_width
            center_norm = center * 32.0 / h  # since LQ is resized to height 32
            width_norm = width * 32.0 / h
            preds_locs[0, 2*i] = center_norm / lq_width
            preds_locs[0, 2*i+1] = width_norm / lq_width

        

        

        '''
        Step 2: Predicting font style w.
        '''
        print(LQ.size())
        with torch.no_grad():
            _, _, w = modelEncoder(LQ)
        
        
        labels = get_labels_from_text(recognized_chars)
        pre_text = recognized_chars

        assert w.size(0) == 1
        w0 = w[:1,...].clone() #

        if manual_label:
            tmp_str = img_basename.split('_')
            pre_text = tmp_str[-1]
            num_given_labels = len(pre_text)
            if num_given_labels != len(labels):
                print(f'Warning!!! The given text has inconsistent number {num_given_labels} with our predicted lables ({num_boxes}). Please double check it.')
            labels = get_labels_from_text(pre_text)
            print('Restoring {}. Using given text: {}'.format(img_name, pre_text))
        else:
            pre_text = get_text_from_labels(labels)
            print('Restoring {}. Using predicted text: {}'.format(img_name, pre_text))


        if len(pre_text) < 1:
            print('Warning!!! No character is detected. Continue...')
            continue



        '''
        Step 3: Generating structure prior.
        '''
        prior_characters = []
        prior_features64 = []
        prior_features32 = []
        labels = torch.Tensor(labels).type(torch.LongTensor).unsqueeze(1)
        try:
            with torch.no_grad():
                prior_cha, prior_fea64, prior_fea32 = modelTSPGAN(styles=w0.repeat(labels.size(0), 1), labels=labels, noise=None)
            prior_characters.append(prior_cha)
            prior_features64.append(prior_fea64)
            prior_features32.append(prior_fea32)
        except:
            traceback.print_exc()
            print('Error in {}. Continue...'.format(img_basename))
            continue
        

        '''
        Step 4: Restoring the LR input.
        '''
        with torch.no_grad():   
            sr_results = modelSR(LQ, prior_features64, prior_features32, preds_locs)
        sr_results = sr_results * 0.5 + 0.5
        sr_results = sr_results.squeeze(0).permute(1, 2, 0).flip(2)
        sr_results = np.clip(sr_results.float().cpu().numpy(), 0, 1) * 255.0
        ShowSR = sr_results[:, :ShowLQ.shape[1], :]


        '''
        Step 5: Showing the SR results.
        '''
        # structure prior
        prior_cha = (prior_cha * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy()
        prior128 = prior_cha[0]
        for i in range(1, len(prior_cha)):
            prior128 = np.hstack((prior128, prior_cha[i]))
        prior = cv2.resize(prior128, (ShowLQ.shape[1], ShowLQ.shape[0])) * 255

        ShowLocs = ShowLQ.copy()
        Locs = preds_locs.clone()
        pad = 2
        img_max_width = 16*128

        # bounding box
        padr = 1
        for c in range(len(pre_text)):
            l = c * 2
            center, width = int(Locs[0][l].item()*img_max_width), int(Locs[0][l+1].item()*img_max_width)
            x = center - width
            y = center + width
            ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 0] = ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 0]*0 + 255
            ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 0] = ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 0]*0 
            ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 1] = ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 1]*0 
            ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 1] = ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 1]*0
            ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 2] = ShowLocs[:64, max(0, x-pad):min(x+pad, img_max_width), 2]*0
            ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 2] = ShowLocs[64:, max(0, y-padr):min(y+padr, img_max_width), 2]*0 + 255
        cv2.imwrite(osp.join(save_path, img_basename+'_{}.png'.format(pre_text)), np.vstack((ShowLQ[:,:,::-1], ShowLocs[:,:,::-1], ShowSR, prior)))
        # exit('ss')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_path', type=str, default='./Testsets/LQs')
    parser.add_argument('-o', '--save_path', type=str, default=None)
    parser.add_argument('-m', '--manual', action='store_true')
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None:
        TIMESTAMP = time.strftime("%m-%d_%H-%M", time.localtime())
        save_path = osp.join(args.test_path+'_'+TIMESTAMP+'_MARCONet')
    os.makedirs(save_path, exist_ok=True)

    print('#'*96)
    print('{:>28s} : {:s}'.format('Input Path', args.test_path))
    print('{:>28s} : {:s}'.format('Save Path', save_path))
    if args.manual:
        print('{:>28s} : {}'.format('The format of text label', 'using given text label (Please DOUBLE CHECK the LR image name)'))
    else:
        print('{:>28s} : {}'.format('The format of text label', 'using predicted text label from modelscope. Updated on Jun 26, 2025'))


    main(args.test_path, save_path, args.manual)


