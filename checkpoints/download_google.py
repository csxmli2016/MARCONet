import gdown

url1 = 'https://drive.google.com/file/d/1doJCuI4DXUJDZ-w5KCjZ2rNY9NASkIg1/view?usp=sharing'
output1 = './tmp/net_prior_generation.pth'
gdown.download(url=url1, output=output1, quiet=False, fuzzy=True)

url2 = 'https://drive.google.com/file/d/1_QvRTuNX2zMXfwIHLuCLG5tBNxaP-x1K/view?usp=sharing'
output2 = './tmp/net_sr.pth'
gdown.download(url=url2, output=output2, quiet=False, fuzzy=True)

url3 = 'https://drive.google.com/file/d/1pkYSo5UH202cvSS3eJJ0xvFA9q-mEX4d/view?usp=sharing'
output3 = './tmp/net_transformer_encoder.pth'
gdown.download(url=url3, output=output3, quiet=False, fuzzy=True)