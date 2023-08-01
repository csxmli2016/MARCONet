import numpy as np
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel

from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F

from tspgan.alphabets import alphabet
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import normalize



import torchvision

@MODEL_REGISTRY.register()
class TSPGANModel(BaseModel):
    def __init__(self, opt):
        super(TSPGANModel, self).__init__(opt)
        self.CommonWords = alphabet
        # 0 define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            print('Successfully load pretrain Generator models!!!!!!!!!!'+load_path)
        self.net_g.train()


        self.TbShowNum = 1
        self.CheckNum = self.opt['datasets']['train'].get('check_num',16)
        self.num_style_feat = opt['network_g']['num_style_feat']

        if self.is_train:
            self.init_training_settings()


    def init_training_settings(self):
        train_opt = self.opt['train']

        #####################################
        ## define models
        #####################################
        # 1 define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)
            print('Successfully load pretrain Discriminator models!!!!!!!!!!'+load_path)
        self.net_d.train()

        # 2 define network net_srd
        self.net_srd = build_network(self.opt['network_srd'])
        self.net_srd = self.model_to_device(self.net_srd)
        self.print_network(self.net_srd)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_srd', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_srd', 'params')
            self.load_network(self.net_srd, load_path, self.opt['path'].get('strict_load_srd', True), param_key)
            print('Successfully load pretrain SRDiscriminator models!!!!!!!!!!'+load_path)
        self.net_srd.train()


        # 3 define encoder 
        self.net_encoder = build_network(self.opt['network_encoder'])
        self.net_encoder = self.model_to_device(self.net_encoder)
        self.print_network(self.net_encoder)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_encoder', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_encoder', 'params')
            self.load_network(self.net_encoder, load_path, self.opt['path'].get('strict_load_encoder', False), param_key)
            print('Successfully load pretrain Encoder models!!!!!!!!!!'+load_path)
        self.net_encoder.train() 


        # 4 define sr network
        self.net_sr = build_network(self.opt['network_sr'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.print_network(self.net_sr)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_sr', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_sr', 'params')
            self.load_network(self.net_sr, load_path, self.opt['path'].get('strict_load_sr', False), param_key)
            print('Successfully load pretrain SR models!!!!!!!!!!'+load_path)
        self.net_sr.train()


        #####################################
        ## define losses
        #####################################
        # 1 reconstruction loss
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        self.cri_srpix = build_loss(train_opt['srpixel_opt']).to(self.device)
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        self.cri_loc = torch.nn.SmoothL1Loss(reduction='mean').to(self.device)
        # 2 gan loss
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        self.cri_ctc = build_loss(train_opt['ctc_opt']).to(self.device)
        self.cri_ce = build_loss(train_opt['ce_opt']).to(self.device)

        # regularization weights
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.path_reg_weight = train_opt['path_reg_weight']  # for generator

        self.net_g_reg_every = train_opt['net_g_reg_every']
        self.net_d_reg_every = train_opt['net_d_reg_every']
        self.mixing_prob = train_opt['mixing_prob']

        self.mean_path_length = 0

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        net_g_reg_ratio = self.net_g_reg_every / (self.net_g_reg_every + 1)
        
        # 0 optimizer srd
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for name, param in self.net_srd.named_parameters():
            normal_params.append(param)
        optim_params_srd = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_srd']['lr']
        }]
        optim_type = train_opt['optim_srd'].pop('type')
        lr = train_opt['optim_srd']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_srd = self.get_optimizer(optim_type, optim_params_srd, lr, betas=betas)
        self.optimizers.append(self.optimizer_srd)

        # 1 optimizer d
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        normal_params = []
        for name, param in self.net_d.named_parameters():
            normal_params.append(param)
        optim_params_d = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]
        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)


        # 2 optimizer prior 
        normal_params = []
        for name, param in self.net_g.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # 3 optimizer encoder
        normal_params = []
        for name, param in self.net_encoder.named_parameters():
            normal_params.append(param)
        optim_params_encoder = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_encoder']['lr']
        }]
        optim_type = train_opt['optim_encoder'].pop('type')
        lr = train_opt['optim_encoder']['lr'] * net_g_reg_ratio # set net_g_reg_ratio same for ocr
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_encoder = self.get_optimizer(optim_type, optim_params_encoder, lr, betas=betas)
        self.optimizers.append(self.optimizer_encoder)

        # 4 optimizer sr 
        normal_params = []
        for name, param in self.net_sr.named_parameters():
            normal_params.append(param)
        optim_params_sr = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_sr']['lr']
        }]
        optim_type = train_opt['optim_sr'].pop('type')
        lr = train_opt['optim_sr']['lr'] * net_g_reg_ratio # set net_g_reg_ratio same for ocr
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_sr = self.get_optimizer(optim_type, optim_params_sr, lr, betas=betas)
        self.optimizers.append(self.optimizer_sr)


    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.mask = data['mask'].to(self.device)

        self.label_gt = data['label'].to(self.device)
        self.lq = data['lq'].to(self.device)
        self.boxinfo = data['boxinfo'].to(self.device)


    def get_text_from_labels(self, TestPreds):
        if TestPreds.dim() > 2:
            PredsInds = torch.max(TestPreds.detach(), 2)[1]
            PredsInds = PredsInds[0]
        elif TestPreds.dim() > 1:
            PredsInds = torch.max(TestPreds.detach(), 1)[1]
        else:
            PredsInds = TestPreds
        PredsText = ''
        for i in range(PredsInds.size(0)):
            if (not (i > 0 and PredsInds[i - 1] == PredsInds[i])) and PredsInds[i] < len(self.CommonWords):
                PredsText = PredsText + self.CommonWords[PredsInds[i]]
        return PredsText

    
    def clear_labels(self, TestPreds):
        labels = []
        if TestPreds.dim() > 2:
            PredsInds = torch.max(TestPreds.detach(), 2)[1]
            PredsInds = PredsInds[0]
        elif TestPreds.dim() > 1:
            PredsInds = torch.max(TestPreds.detach(), 1)[1]
        else:
            PredsInds = TestPreds
        for i in range(PredsInds.size(0)):
            if (not (i > 0 and PredsInds[i - 1] == PredsInds[i])) and PredsInds[i] < len(alphabet):
                labels.append(PredsInds[i])
        return labels
    
    def get_current_visuals(self):
        ShowNum = min(self.TbShowNum, self.gt.size(0))
        ShowGT = self.gt[:ShowNum,:,:,:].detach()
        ShowLQ = F.interpolate(self.lq[:ShowNum,:,:,:].detach(), (ShowGT.size(2), ShowGT.size(3)), mode='bilinear', align_corners=False)
        ShowLQSR = self.sr_results[:ShowNum,:,:,:].detach()

        img_max_width = self.CheckNum*128

        b = 0
        TestPreds = self.TestPreds
        if TestPreds.dim() > 2:
            PredsInds = torch.max(TestPreds.detach(), 2)[1]
            PredsInds = PredsInds[0]
        elif TestPreds.dim() > 1:
            PredsInds = torch.max(TestPreds.detach(), 1)[1]
        else:
            PredsInds = TestPreds
        PredsText = ''
        for i in range(PredsInds.size(0)):
            if (not (i > 0 and PredsInds[i - 1] == PredsInds[i])) and PredsInds[i] < len(self.CommonWords):
                PredsText = PredsText + self.CommonWords[PredsInds[i]]

        bgcolor = (0,0,0)
        CImg = Image.new('RGB', (512, 32), bgcolor)
        drawer = ImageDraw.Draw(CImg)

        drawer.text((10,-10), PredsText, font=ImageFont.truetype('./TrainData/FontsType-V1/msyhbd.ttc', 32, encoding="utf-8"), fill=(0,255,0))
        NImg = np.array(CImg).astype(np.float32) / 255.0 # RGB 0~1 CHW
        NImg = torch.from_numpy(NImg) 
        NImg = NImg.permute((2,0,1)).unsqueeze(0)
        normalize(NImg, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True)
        ShowPreText = NImg

        ShowPredLocs = ShowLQ.clone()
        ShowGTLocs = ShowGT.clone()


        Locs = self.TestPreds_locs[:ShowNum,...] # center, width, 2048
        GTLocs = self.boxinfo[:ShowNum,...]
        pad, padGT = 2, 1
        padr = 1
        
        for b in range(Locs.size(0)):
            for l in range(0, Locs.size(1), 2):
                center, width = int(Locs[b][l].item()), int(Locs[b][l+1].item())
                x = center - width
                y = center + width
                ShowPredLocs[b, 0, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowPredLocs[b, 0, :64, max(0, x-pad):min(x+pad, img_max_width)]*0 + 1
                ShowPredLocs[b, 0, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowPredLocs[b, 0, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0 
                ShowPredLocs[b, 1, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowPredLocs[b, 1, :64, max(0, x-pad):min(x+pad, img_max_width)]*0 
                ShowPredLocs[b, 1, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowPredLocs[b, 1, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0
                ShowPredLocs[b, 2, :64, max(0, x-pad):min(x+pad, img_max_width)] = ShowPredLocs[b, 2, :64, max(0, x-pad):min(x+pad, img_max_width)]*0
                ShowPredLocs[b, 2, 64:, max(0, y-padr):min(y+padr, img_max_width)] = ShowPredLocs[b, 2, 64:, max(0, y-padr):min(y+padr, img_max_width)]*0 + 1

                
                x, y = int(GTLocs[b][l].item()*img_max_width), int(GTLocs[b][l+1].item()*img_max_width)
                ShowGTLocs[b, 0, :, max(0, x-padGT):min(x+padGT, img_max_width)] = ShowGTLocs[b, 0, :, max(0, x-padGT):min(x+padGT, img_max_width)]*0
                ShowGTLocs[b, 0, :, max(0, y-padGT):min(y+padGT, img_max_width)] = ShowGTLocs[b, 0, :, max(0, y-padGT):min(y+padGT, img_max_width)]*0
                ShowGTLocs[b, 1, :, max(0, x-padGT):min(x+padGT, img_max_width)] = ShowGTLocs[b, 1, :, max(0, x-padGT):min(x+padGT, img_max_width)]*0 + 1
                ShowGTLocs[b, 1, :, max(0, y-padGT):min(y+padGT, img_max_width)] = ShowGTLocs[b, 1, :, max(0, y-padGT):min(y+padGT, img_max_width)]*0 + 1
                ShowGTLocs[b, 2, :, max(0, x-padGT):min(x+padGT, img_max_width)] = ShowGTLocs[b, 2, :, max(0, x-padGT):min(x+padGT, img_max_width)]*0
                ShowGTLocs[b, 2, :, max(0, y-padGT):min(y+padGT, img_max_width)] = ShowGTLocs[b, 2, :, max(0, y-padGT):min(y+padGT, img_max_width)]*0
        

        ShowCharacterGT = self.gt_characters[:,...].clone().detach()
        ShowCharacterSR = self.prior_characters128[:,...].clone().detach()

        return {'1_ShowGT':ShowGT, '1_ShowLQSR':ShowLQSR, '1_ShowTestPreText': ShowPreText, 
                '2_ShowTestPreLocs': ShowPredLocs,
                '3_ShowPrior64': self.prior_characters64[:16,...].detach(), 
                '7_ShowCharacterGT': ShowCharacterGT[:16,...], '7_ShowCharacterSR': ShowCharacterSR[:16,...], 
                '6_ShowPCSR':self.tmp_sr_chars[:16,...], '6_ShowPCGT':self.tmp_gt_chars[:16,...],
                }


    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        l_g_total = 0
        
        for p in self.net_d.parameters():
            p.requires_grad = False
        for p in self.net_srd.parameters():
            p.requires_grad = False
        

        # optimize net_encoder
        self.optimizer_encoder.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_sr.zero_grad()
        preds_cls, preds_locs_l_r, w = self.net_encoder(self.lq)

        preds_locs = preds_locs_l_r.clone()
        for b in range(self.boxinfo.size(0)):
            for n in range(0, self.boxinfo.size(1), 2):
                preds_locs[b][n] = (preds_locs_l_r[b][n+1] + preds_locs_l_r[b][n]) / 2.0 #center location
                preds_locs[b][n+1] = (preds_locs_l_r[b][n+1] - preds_locs_l_r[b][n]) / 2.0 # width


        self.TestPreds = preds_cls.clone().detach()
        preds_locs_2048 = preds_locs.clone() * 2048
        self.TestPreds_locs = preds_locs_2048.clone().detach() #


        l_ctc = self.cri_ctc(preds_cls, self.label_gt) * self.opt['train'].get('ctc_loss_lambda',1)
        if torch.isnan(l_ctc).sum() > 0:
            print('l_ctc NAN')
        l_g_total += l_ctc
        loss_dict['l_ctc'] = l_ctc


        #
        relative_box = self.boxinfo.clone().detach() #
        for b in range(self.boxinfo.size(0)):
            for n in range(0, self.boxinfo.size(1), 2):
                relative_box[b][n] = (self.boxinfo[b][n+1] + self.boxinfo[b][n]) / 2.0 #center location
                relative_box[b][n+1] = (self.boxinfo[b][n+1] - self.boxinfo[b][n]) / 2.0 # width


        preds_center = []
        gt_center = []
 
        for b in range(self.boxinfo.size(0)):
            for n in range(0, self.boxinfo.size(1), 2):
                preds_center.append(preds_locs_2048[b][n]) #only center
                gt_center.append((self.boxinfo[b][n+1] + self.boxinfo[b][n]) / 2.0 * 2048)
        preds_center = torch.stack(preds_center)
        gt_center = torch.stack(gt_center)
        l_loc_center = self.cri_loc(preds_center, gt_center)  * self.opt['train'].get('loc_loss_lambda',1) *2
        l_g_total += l_loc_center
        loss_dict['l_loc_center'] = l_loc_center

        l_loc = self.cri_loc(preds_locs_l_r*2048, self.boxinfo*2048) * self.opt['train'].get('loc_loss_lambda',1) #
        if torch.isnan(l_loc).sum() > 0:
            print('l_loc NAN')
        l_g_total += l_loc
        loss_dict['l_loc'] = l_loc


        loc_diff = []
        loc_iou = []
        for b in range(self.boxinfo.size(0)):
            for n in range(0, self.boxinfo.size(1), 2):
                if self.boxinfo[b][n+1] - self.boxinfo[b][n] > 0.0:
                    diff = preds_locs_2048[b][n+1] #width
                    loc_diff.append(diff)

                    #compute IOU loss
                    x1 = preds_locs_2048[b][n] - preds_locs_2048[b][n+1] 
                    x2 = preds_locs_2048[b][n] + preds_locs_2048[b][n+1] 
                    x1g = (relative_box[b][n] - relative_box[b][n+1])*2048
                    x2g = (relative_box[b][n] + relative_box[b][n+1])*2048 
                    xkis1 = torch.max(x1, x1g)
                    xkis2 = torch.min(x2, x2g)
                    inter_area = torch.max(xkis2-xkis1, torch.zeros_like(x1))
                    union_area = x2-x1 + x2g-x1g - inter_area
                    iou = 1 - inter_area / torch.clamp(union_area, min=1e-6)
                    loc_iou.append(iou)

        self.Tmp_diff = loc_diff
        self.Tmp_iou = loc_iou
        self.Tmp_preds = preds_locs_2048
        self.Tmp_gtloc = relative_box*2048


        
        if len(loc_iou) > 0:
            loc_iou = torch.stack(loc_iou)
            l_loc_iou = loc_iou.mean()  * self.opt['train'].get('iou_loss_lambda',1)
            if torch.isnan(l_loc_iou).sum() > 0:
                print('l_loc_iou NAN')
            l_g_total += l_loc_iou 
            loss_dict['l_loc_iou'] = l_loc_iou


        prior_characters = []
        gt_characters = []
        label_characters = []
        prior_features64 = []
        prior_features32 = []
        prior_rgb64 = []
        prior_rgb32 = []

        
        nobg_gt = self.mask * 2 - 1 #-1~1
        for b in range(w.size(0)):
            w0 = w[b:b+1,...].clone() #1*512
            boxinfo = self.boxinfo[b].clone().detach() * self.gt.size(-1)
            label = []
            for n in range(0, boxinfo.size(0), 2):
                if boxinfo[n+1] - boxinfo[n] > 0.0 and self.label_gt[b][n//2] != 6735:
                    label.append(self.label_gt[b][n//2])
                    gt_char = nobg_gt[b:b+1, :, :, int(boxinfo[n].item()):int(boxinfo[n+1].item())]
                    if gt_char.size(3) > 128:
                        gt_char = F.interpolate(gt_char, (128,128), mode='bilinear', align_corners=False)
                    nobg_mask = torch.zeros(1,3,128,128).to(self.device) - 1
                    width = gt_char.size(3)//2
                    nobg_mask[:, :, :, 64-width:64-width+gt_char.size(3)] = gt_char.clone()
                    gt_characters.append(nobg_mask)
                
            label = torch.Tensor(label).type(torch.LongTensor).view(-1, 1).to(self.device)
            prior_cha, prior_fea64, prior_fea32, rgb64, rgb32 = self.net_g(styles=w0.repeat(label.size(0), 1), labels=label, noise=None) #b *n * w * h
            prior_characters.append(prior_cha)
            prior_features64.append(prior_fea64)
            prior_features32.append(prior_fea32)
            prior_rgb64.append(rgb64)
            prior_rgb32.append(rgb32)
            label_characters.append(label)
        
        self.prior_characters128 = torch.cat(prior_characters, dim=0)
        self.prior_characters64 = torch.cat(prior_rgb64, dim=0)
        self.prior_characters32 = torch.cat(prior_rgb32, dim=0)
        self.gt_characters = torch.cat(gt_characters, dim=0)
        self.label_characters = torch.cat(label_characters, dim=0)#.view(-1,1)
        
        
    
        # optimize net_g and net_encoder 
        l_g_pix128 = self.cri_pix(self.prior_characters128, self.gt_characters) * self.opt['train'].get('pixel_loss_lambda128',1) # -1 ~ 1

        inter_area = (self.prior_characters128 + 1)/2 * (self.gt_characters + 1 ) / 2 # 0~1
        union_area = (self.prior_characters128 + 1)/2 + (self.gt_characters + 1 ) / 2 - inter_area
        l_g_iou128 = (1 - inter_area / torch.clamp(union_area, min=1e-6)).mean() * self.opt['train'].get('pixel_loss_iou',2)

        if torch.isnan(l_g_pix128).sum() > 0:
            print('l_g_pix128 NAN')
        l_g_total += l_g_pix128
        loss_dict['l_g_pix128'] = l_g_pix128

        if torch.isnan(l_g_iou128).sum() > 0: 
            print('l_g_iou128 NAN')
        l_g_total += l_g_iou128
        loss_dict['l_g_iou128'] = l_g_iou128


        l_g_pix64 = self.cri_pix(self.prior_characters64, F.interpolate(self.gt_characters, (64,64), mode='bilinear', align_corners=False)) * self.opt['train'].get('pixel_loss_lambda64',2) 
        if torch.isnan(l_g_pix64).sum() > 0:
            print('l_g_pix64 NAN')
        l_g_total += l_g_pix64
        loss_dict['l_g_pix64'] = l_g_pix64

        l_g_pix32 = self.cri_pix(self.prior_characters32, F.interpolate(self.gt_characters, (32,32), mode='bilinear', align_corners=False)) * self.opt['train'].get('pixel_loss_lambda32',4) 
        if torch.isnan(l_g_pix32).sum() > 0:
            print('l_g_pix32 NAN')
        l_g_total += l_g_pix32 
        loss_dict['l_g_pix32'] = l_g_pix32
        
        d_sr_characters = self.prior_characters128.clone()


        fake_pred = self.net_d(d_sr_characters)
        l_g_gan = self.cri_gan(fake_pred, True, is_disc=False) * self.opt['train'].get('gan_loss_lambda',1) 
        if torch.isnan(l_g_gan).sum() > 0:
            print('l_g_gan NAN')
        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan


        ### optimize for super-resolution
        self.sr_results = self.net_sr(self.lq, prior_features64, prior_features32, preds_locs) #
        l_sr_pix = self.cri_srpix(self.sr_results, self.gt) 

        l_g_total += l_sr_pix 
        loss_dict['l_sr_pix'] = l_sr_pix

        sr_result_patches = []
        gt_patches = []

        ### this section is for cropping without alignment
        for b in range(self.lq.size(0)):
            boxinfo = self.boxinfo[b].cpu().detach() * self.gt.size(-1)
            max_length = 0
            for n in range(0, boxinfo.size(0), 2):
                if boxinfo[n+1] - boxinfo[n] > 0.0 and max_length < boxinfo[n+1]:
                    max_length = boxinfo[n+1].detach().int().item()
            patch_num = max_length // 128 + 1
            for p in range(patch_num):
                sr_result_patches.append(self.sr_results[b:b+1, :, :, 128*p:128*(p+1)].clone())
                gt_patches.append(self.gt[b:b+1, :, :, 128*p:128*(p+1)].clone())   
        sr_result_patches = torch.cat(sr_result_patches, dim=0)
        gt_patches = torch.cat(gt_patches, dim=0)


        sr_result_chars = []
        gt_chars = []
        for b in range(self.lq.size(0)):
            boxinfo = self.boxinfo[b].clone().detach() * self.gt.size(-1)
            for n in range(0, boxinfo.size(0), 2):
                if boxinfo[n+1] - boxinfo[n] > 0.0 and self.label_gt[b][n//2] != 6735:
                    center = (boxinfo[n+1] + boxinfo[n]) / 2
                    center = center.int()
                    if center < 64:
                        x1 = 0
                    else:
                        x1 = center - 64
                    if center + 64 > self.gt.size(-1):
                        x2 = self.gt.size(-1)
                    else:
                        x2 = center + 64
                    sr_char = self.sr_results[b:b+1, :, :, x1:x2].clone()
                    gt_char = self.gt[b:b+1, :, :, x1:x2].clone()
                    if sr_char.size(-1) < 128:
                        sr_char = F.interpolate(sr_char, (128,128), mode='bilinear')
                        gt_char = F.interpolate(gt_char, (128,128), mode='bilinear')
                    sr_result_chars.append(sr_char)
                    gt_chars.append(gt_char)   

        sr_result_chars = torch.cat(sr_result_chars, dim=0)
        gt_chars = torch.cat(gt_chars, dim=0)


        self.tmp_sr_chars = sr_result_chars.clone()
        self.tmp_gt_chars = gt_chars.clone()
        self.tmp_sr_patches = sr_result_patches.clone()
        self.tmp_gt_patches = gt_patches.clone()

        fake_srpred = self.net_srd(torch.cat((sr_result_chars, self.prior_characters128.detach()), dim=1))
        l_sr_gan = self.cri_gan(fake_srpred, True, is_disc=False) * self.opt['train'].get('srgan_loss_lambda',1) 

        l_g_total += l_sr_gan 
        loss_dict['l_sr_d_pr'] = l_sr_gan

        fake_pred = self.net_d(sr_result_chars)
        l_sr_rgan = self.cri_gan(fake_pred, True, is_disc=False) * self.opt['train'].get('gan_loss_lambda',1) 

        l_g_total += l_sr_rgan
        loss_dict['l_sr_d_r'] = l_sr_rgan


        l_sr_percep = self.cri_perceptual(sr_result_patches, gt_patches)
 
        l_g_total += l_sr_percep
        loss_dict['l_sr_percep'] = l_sr_percep


        l_g_total.backward()

        self.optimizer_encoder.step()
        self.optimizer_g.step()
        self.optimizer_sr.step()

        #############################################################
        ### optimize net_d
        #############################################################
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        fake_pred = self.net_d(sr_result_chars.detach())
        real_pred = self.net_d(gt_chars)
        l_d = self.cri_gan(real_pred, True, is_disc=True) + self.cri_gan(fake_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        l_d.backward()
        self.optimizer_d.step()
        

        # optimize net_srd
        for p in self.net_srd.parameters():
            p.requires_grad = True
        self.optimizer_srd.zero_grad()
        fake_pred = self.net_srd(torch.cat((sr_result_chars.detach(), self.prior_characters128.detach()), dim=1))
        real_pred = self.net_srd(torch.cat((gt_chars, self.gt_characters), dim=1))
        l_srd = self.cri_gan(real_pred, True, is_disc=True) + self.cri_gan(fake_pred, False, is_disc=True)
        loss_dict['l_srd'] = l_srd
        l_srd.backward()
        self.optimizer_srd.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        
        
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        print('begin to test')
        vis_vars = self.get_current_visuals()
        for label, images in vis_vars.items():
            images = (images + 1)/2.0
            grid = torchvision.utils.make_grid(images,normalize=False, scale_each=True)
            tb_logger.add_image(label, grid, current_iter)
    
    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.net_srd, 'net_srd', current_iter)
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_encoder, 'net_encoder', current_iter)
        self.save_training_state(epoch, current_iter)
