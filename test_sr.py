
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



def clear_labels(TestPreds):
    labels = []
    PredsInds = torch.max(TestPreds.detach(), 1)[1]
    for i in range(PredsInds.size(0)):
        if (not (i > 0 and PredsInds[i - 1] == PredsInds[i])) and PredsInds[i] < len(alphabet):
            labels.append(PredsInds[i])
    return labels

def main(L_path, save_path, manual_label, use_real_ocr, use_new_bbox):
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

    if use_new_bbox:
        modelBBox = ocr.TransformerOCR(use_new_bbox=True)
        modelBBox.load_state_dict(torch.load('./checkpoints/net_new_bbox.pth')['params'], strict=True)
        modelBBox.eval()
        modelBBox = modelBBox.to(device)
        print('{:>28s} : {} M Parameters'.format('New BBOX Network', print_networks(modelBBox)))

    if use_real_ocr:
        modelOCR = ocr.TransformerOCR()
        modelOCR.load_state_dict(torch.load('./checkpoints/net_real_world_ocr.pth')['params'], strict=True)
        modelOCR.eval()
        modelOCR = modelOCR.to(device)
        print('{:>28s} : {} M Parameters'.format('New Real-world OCR Network', print_networks(modelOCR)))

    print('{:>28s} : {} M Parameters'.format('Transformer Encoder', print_networks(modelEncoder)))
    print('{:>28s} : {} M Parameters'.format('Structure Prior Network', print_networks(modelTSPGAN)))
    print('{:>28s} : {} M Parameters'.format('Super-Resolution Network', print_networks(modelSR)))
    
    
    print('#'*64)
    
    modelTSPGAN = modelTSPGAN.to(device)
    modelSR = modelSR.to(device)
    modelEncoder = modelEncoder.to(device)
    
    

    torch.cuda.empty_cache()
    
    img_names = os.listdir(L_path)
    img_names.sort()
    for img_name in img_names:
        '''
        Step 1: Reading Image
        '''
        img_path = osp.join(L_path, img_name)
        img_basename, ext = osp.splitext(osp.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        ShowLQ = cv2.resize(img, (0,0), fx=128/h, fy=128/h,  interpolation=cv2.INTER_CUBIC)
        LQ = cv2.resize(img, (0,0), fx=32/h, fy=32/h,  interpolation=cv2.INTER_CUBIC)
        ori_lq_w = LQ.shape[1]

        TextLQFillBG = np.zeros((32, 32*16, 3)).astype(LQ.dtype)
        if LQ.shape[-2] <= 32*16:
            TextLQFillBG[:, :LQ.shape[-2], :] = TextLQFillBG[:, :LQ.shape[-2], :] + LQ
            LQ = TextLQFillBG
        else:
            print(['\tLQ width is not normal... The width is larger than 512', LQ.shape])
            continue

        LQ = transforms.ToTensor()(LQ)
        LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)

        LQ = LQ.unsqueeze(0)
        LQ = LQ.to(device)

        '''
        Step 2: Predicting the character labels, bounding boxes and font style.
        '''
        with torch.no_grad():
            preds_cls, preds_locs_l_r, w = modelEncoder(LQ)
        
        labels = clear_labels(preds_cls[0])
        pre_text = get_text_from_labels(labels)

        preds_locs = preds_locs_l_r.clone()
        for n in range(0, 16*2, 2):
            preds_locs[0][n] = (preds_locs_l_r[0][n+1] + preds_locs_l_r[0][n]) / 2.0 #center
            preds_locs[0][n+1] = (preds_locs_l_r[0][n+1] - preds_locs_l_r[0][n]) / 2.0 # width
        


        assert w.size(0) == 1
        w0 = w[:1,...].clone() #

        '''
        Step 2.5: Predicting the character labels using real-world OCR model trained on real-world chinese dataset, see:
        https://github.com/FudanVI/benchmarking-chinese-text-recognition/tree/main
        '''
        
        if use_real_ocr:
            LQForOCR = cv2.resize(img, (256,32), interpolation=cv2.INTER_CUBIC)
            LQForOCR = transforms.ToTensor()(LQForOCR)
            LQForOCR = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQForOCR)

            LQForOCR = LQForOCR.unsqueeze(0)
            LQForOCR = LQForOCR.to(device)

            #---------------character classification--------------------
            max_length = 20
            batch = 1
            pred = torch.zeros(batch,1).long().cuda()
            image_features = None
            prob = torch.zeros(batch, max_length).float()
            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                with torch.no_grad():
                    result = modelOCR(image=LQForOCR, text_length=length_tmp, text_input=pred, conv_feature=image_features, test=True)
                prediction = result['pred']
                now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
                prob[:,i] = torch.max(torch.softmax(prediction,2), 2)[0][:,-1]
                pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
                image_features = result['conv']

            text_pred_list = []
            now_pred = []
            for j in range(max_length):
                if pred[0][j] != 6737:
                    now_pred.append(pred[0][j])
                else:
                    break
            text_pred_list = torch.Tensor(now_pred)[1:].long().cuda()
            pre_text = ""
            for i in text_pred_list:
                if i == (len(alphabet)+2):
                    continue
                pre_text += alphabet[i-2]

            labels = get_labels_from_text(pre_text)


        

        '''
        Step 2.75: Predicting the bbox using our synthtic images
        '''
        if use_new_bbox:
            LQForBBox = cv2.resize(img, (256,32), interpolation=cv2.INTER_CUBIC)
            LQForBBox = transforms.ToTensor()(LQForBBox)
            LQForBBox = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQForBBox)

            LQForBBox = LQForBBox.unsqueeze(0)
            LQForBBox = LQForBBox.to(device)

            #---------------character classification--------------------
            max_length = 20
            batch = 1
            pred = torch.zeros(batch,1).long().cuda()
            loc = torch.zeros(batch,1).float().cuda()
            image_features = None
            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                with torch.no_grad():
                    result = modelBBox(image=LQForBBox, text_length=length_tmp, text_input=pred, conv_feature=image_features, test=True)
                prediction = result['pred']
                now_pred = torch.max(torch.softmax(prediction,2), 2)[1]
                pred = torch.cat((pred, now_pred[:,-1].view(-1,1)), 1)
                now_loc = result['loc'][:,-1].view(-1,1) #* self.opt['datasets']['train']['ocr_width'] # using sigmoid, from 0~1 to 0~256
                loc = torch.cat((loc, now_loc), 1)
                image_features = result['conv']

            text_pred_list_bbox = []
            now_pred = []
            for j in range(max_length):
                if pred[0][j] != 6737:
                    now_pred.append(pred[0][j])
                else:
                    break
            text_pred_list_bbox = torch.Tensor(now_pred)[1:].long().cuda()
            pre_text_bbox = ""
            for i in text_pred_list_bbox:
                if i == (len(alphabet)+2):
                    continue
                pre_text_bbox += alphabet[i-2]

            if len(pre_text_bbox) != len(pre_text):
                print('!!!!!! Change the label from {} to {}'.format(pre_text, pre_text_bbox))
                pre_text = pre_text_bbox
                labels = get_labels_from_text(pre_text)

            preds_locs = preds_locs_l_r.clone()
            for n in range(0, 16*2, 2):
                preds_locs[0][n] = int(loc[0][n//2+2].item()) * ori_lq_w / 256 / 512 # for ocr 32*512
                preds_locs[0][n+1] = 0 

        if manual_label:
            tmp_str = img_basename.split('_')
            pre_text = tmp_str[-1]
            if len(pre_text) != len(labels):
                print('\t !!!The given text has inconsistent number with our predicted lables. Please double check it.')
            labels = get_labels_from_text(pre_text)
            print('Restoring {}. The given text: {}'.format(img_name, pre_text))
        else:
            pre_text = get_text_from_labels(labels)
            print('Restoring {}. The predicted text: {}'.format(img_name, pre_text))


        if len(pre_text) > 16:
            print('\tToo much characters. The max length is 16.')
            continue

        if len(pre_text) < 1:
            print('\tNo character is detected. Continue...')
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
            print('\tError in {}. Continue...'.format(img_basename))
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

    '''
    We add new real-world ocr model and new robust bbox detection.
    Set 
    '''
    use_new_bbox = True
    use_real_ocr = True

    save_path = args.save_path
    if save_path is None:
        TIMESTAMP = time.strftime("%m-%d_%H-%M", time.localtime())
        save_path = osp.join(args.test_path+'_'+TIMESTAMP+'_MARCONet')
    os.makedirs(save_path, exist_ok=True)
    print('#'*64)
    print('{:>28s} : {:s}'.format('Input Path', args.test_path))
    print('{:>28s} : {:s}'.format('Save Path', save_path))
    if args.manual:
        print('{:>28s} : {}'.format('The format of text label', 'using given text label (Please DOUBLE CHECK the LR image name)'))
    else:
        print('{:>28s} : {}'.format('The format of text label', 'using predicted text label'))
    
    if use_real_ocr:
        print('{:>28s} : {}'.format('OCR Module', 'using ocr model trained on public chinese ocr dataset (Preferred)'))
    else:
        print('{:>28s} : {}'.format('OCR Module', 'using ocr model trained on our synthetic data'))

    
    

    main(args.test_path, save_path, args.manual, use_real_ocr, use_new_bbox)


