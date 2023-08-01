## Training Code for [Learning Generative Structure Prior for Blind Text Image Super-resolution](https://arxiv.org/pdf/2303.14726.pdf)

[Xiaoming Li](https://csxmli2016.github.io/), [Wangmeng Zuo](https://scholar.google.com/citations?hl=en&user=rUOpCEYAAAAJ&view_op=list_works), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University


## Getting Start
> I use torch 1.12.1 and cuda 11.3
```
conda create -n marcotrain python=3.8 -y
conda activate marcotrain
pip install -r requirements.txt
BASICSR_EXT=True pip install basicsr
```
> Please carefully follow the installation steps, especially the final one with **BASICSR_EXT=True**. 

## Downloading the initial models and fonts
Download the pre-trained models
```
python scripts/download.py
```

If you want to use the Chinese Corpus, you can download them from [BaiduNetDisk](https://pan.baidu.com/s/177ggwkQ-7-vHW3YK2RBYsw?pwd=xxtd)

It contains:
- Baike (1.36 GB)
- News (8.03 GB)
- Wike (1.2 GB)

You can download them and put them into ```./TrainData/ChineseCorpus```, and modify the corpurs_path in Lines 14~16 in ```./options/train.yml```.
> You can use one of them in ```corpus_path1```, and keep ```corpus_path2``` and ```corpus_path3``` None.

## Generating the background images
You can refer to the cropped samples in ```./TrainData/BGSample``` (400*400).

You can generate them by running:
```
python crop_DF2K.py
``` 



## Training the whole framework
```
CUDA_VISIBLE_DEVICES=0 python tspgan/train.py -opt options/train.yml
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch —nproc_per_node=4 —master_port=43210 tspgan/train.py -opt options/train.yml —launcher pytorch
```

## A simple demo for synthesizing the training Data
```
python syndata_demo.py
```

The synthetic results include: 1) LR input, 2) GT, 3) Bounding Box, 4) Prior Image:

<img src="./syn_data_samples/08-01-18_15_01_lq.png"  width="960px">
<img src="./syn_data_samples/08-01-18_15_01_gt.png"  width="960px">
<img src="./syn_data_samples/08-01-18_15_01_locs.png"  width="960px">
<img src="./syn_data_samples/08-01-18_15_01_mask.png"  width="960px">


## License
This project is licensed under <a rel="license" href="https://github.com/csxmli2016/MARCONet/blob/main/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR) and [KAIR](https://github.com/cszn/KAIR).


## Citation

```
@InProceedings{li2023marconet,
author = {Li, Xiaoming and Zuo, Wangmeng and Loy, Chen Change},
title = {Learning Generative Structure Prior for Blind Text Image Super-resolution},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```


