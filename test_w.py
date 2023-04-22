
import torch
import cv2
import numpy as np
import os.path as osp
from models import networks
import torchvision.transforms as transforms
from utils.alphabets import alphabet
import imageio
import argparse


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

def main(w1_path, w2_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelTSPGAN = networks.TSPGAN()
    modelTSPGAN.load_state_dict(torch.load('./checkpoints/net_prior_generation.pth')['params'], strict=True)
    modelTSPGAN.eval()
    modelEncoder = networks.TextContextEncoderV2()
    modelEncoder.load_state_dict(torch.load('./checkpoints/net_transformer_encoder.pth')['params'], strict=True)
    modelEncoder.eval()
    modelTSPGAN = modelTSPGAN.to(device)
    modelEncoder = modelEncoder.to(device)

    torch.cuda.empty_cache()
    
    '''
    Load w1 from LR image 1
    '''
    img1 = cv2.imread(w1_path, cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    h, w, c = img1.shape
    ShowLQ1 = cv2.resize(img1, (0,0), fx=128/h, fy=128/h,  interpolation=cv2.INTER_CUBIC)
    LQ1 = cv2.resize(img1, (0,0), fx=32/h, fy=32/h,  interpolation=cv2.INTER_CUBIC)
    TextLQFillBG1 = np.zeros((32, 32*16, 3)).astype(LQ1.dtype)
    if LQ1.shape[-2] <= 32*16:
        TextLQFillBG1[:, :LQ1.shape[-2], :] = TextLQFillBG1[:, :LQ1.shape[-2], :] + LQ1
        LQ1 = TextLQFillBG1
    else:
        exit(['\tLQ1 width is not normal... The width is larger than 512', LQ1.shape])
    LQ1 = transforms.ToTensor()(LQ1)
    LQ1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ1)
    LQ1 = LQ1.unsqueeze(0)
    LQ1 = LQ1.to(device)

    '''
    Load w2 from LR image 2
    '''
    img2 = cv2.imread(w2_path, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    h, w, c = img2.shape
    ShowLQ2 = cv2.resize(img2, (0,0), fx=128/h, fy=128/h,  interpolation=cv2.INTER_CUBIC)
    LQ2 = cv2.resize(img2, (0,0), fx=32/h, fy=32/h,  interpolation=cv2.INTER_CUBIC)
    TextLQFillBG2 = np.zeros((32, 32*16, 3)).astype(LQ2.dtype)
    if LQ2.shape[-2] <= 32*16:
        TextLQFillBG2[:, :LQ2.shape[-2], :] = TextLQFillBG2[:, :LQ2.shape[-2], :] + LQ2
        LQ2 = TextLQFillBG2
    else:
        exit(['\tLQ2 width is not normal... The width is larger than 512', LQ2.shape])

    LQ2 = transforms.ToTensor()(LQ2)
    LQ2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ2)
    LQ2 = LQ2.unsqueeze(0)
    LQ2 = LQ2.to(device)

    with torch.no_grad():
        preds_cls1, _, w1 = modelEncoder(LQ1)
        preds_cls2, _, w2 = modelEncoder(LQ2)
    ##show the interpolation w on LR image 1
    labels1 = clear_labels(preds_cls1[0])
    labels1 = torch.Tensor(labels1).type(torch.LongTensor).unsqueeze(1)

    with torch.no_grad():
        buff = []
        for i in range(11):
            scale = i / 10
            print('Interpolating w1 and w2 with weight {:.2f}'.format(scale))
            new_w = w1 * scale + w2 * (1 - scale)
            prior_cha, _, _ = modelTSPGAN(styles=new_w.repeat(labels1.size(0), 1), labels=labels1, noise=None)
            prior_cha = (prior_cha * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy()
            prior128 = prior_cha[0]
            for i in range(1, len(prior_cha)):
                prior128 = np.hstack((prior128, prior_cha[i]))
            buff.append((prior128*255.0).astype(np.uint8))
            cv2.imwrite(osp.join(save_path, 'w_{:.2f}.png'.format(scale)), prior128*255.0)
    imageio.mimsave(osp.join(save_path, 'w.gif'), buff,'GIF',duration=0.1)

    print('Finishing interpolation.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w1', '--w1_path', type=str, default='./Testsets/TestW/w1.png')
    parser.add_argument('-w2', '--w2_path', type=str, default='./Testsets/TestW/w2.png')
    parser.add_argument('-o', '--save_path', type=str, default='./Testsets/TestW')
    args = parser.parse_args()

    print('#'*64)
    print('{:>16s} : {:s}'.format('Input w1', args.w1_path))
    print('{:>16s} : {:s}'.format('Input w2', args.w2_path))
    print('{:>16s} : {:s}'.format('Save Path', args.save_path))
    print('#'*64)

    main(args.w1_path, args.w2_path, args.save_path)


