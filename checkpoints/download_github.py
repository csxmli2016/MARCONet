from basicsr.utils.download_util import load_file_from_url

# For Chinese
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_prior_generation.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_sr.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_transformer_encoder.pth', model_dir='./checkpoints')


# For English and Number
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/Eng/net_prior_generation_Eng.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/Eng/net_sr_Eng.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/Eng/net_transformer_encoder_Eng.pth', model_dir='./checkpoints')
