
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class ValDataset(data.Dataset):
    def __init__(self, opt):
        super(ValDataset, self).__init__()
        self.opt = opt

    def __getitem__(self, index):
        
        return {'gt': torch.randn(3,256,256), 'lq': torch.randn(3,256,256), 'lq_path':'./val.png'} # 

    def __len__(self):
        return 1