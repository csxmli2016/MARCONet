from basicsr.utils.download_util import load_file_from_url
from zipfile import ZipFile
import os


print('1. Downloading the initial models....')
os.makedirs('./experiments/init', exist_ok=True)
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_prior_generation.pth', model_dir='./experiments/init/')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_sr.pth', model_dir='./experiments/init/')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_transformer_encoder.pth', model_dir='././experiments/init/')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_srd.pth', model_dir='./experiments/init/')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_d.pth', model_dir='./experiments/init/')


print('2. Downloading the font....')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/FontsType-V1.zip', model_dir='./TrainData/')
with ZipFile('./TrainData/FontsType-V1.zip', 'r') as zObject:
    zObject.extractall(
        path="./TrainData/")
os.remove('./TrainData/FontsType-V1.zip')