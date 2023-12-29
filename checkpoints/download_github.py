from basicsr.utils.download_util import load_file_from_url


load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_prior_generation.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_sr.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_transformer_encoder.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_new_bbox.pth', model_dir='./checkpoints')
load_file_from_url('https://github.com/csxmli2016/MARCONet/releases/download/v1/net_real_world_ocr.pth', model_dir='./checkpoints')
