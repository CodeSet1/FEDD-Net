# FEDD-Net: a frequency diagonal feature enhanced dual-branch diffusion network for low-light image enhancement
### [Paper]() | [Code](https://github.com/CodeSet1/GDRD-Net)

This code is provided solely for academic research purposes and is exclusively restricted from any commercial use.

## 1. Create Environment
- Create Conda Environment
```
conda env create -f environment.yaml
```
- Activate Conda Environment
```
conda activate fedd_net_env
```
We strongly recommend using the configurations provided in the yaml file, as different versions of dependency packages may produce varying results.

## 2. Prepare Your Dataset
You can refer to [LOLv1](https://daooshee.github.io/BMVC2018website/), [LOLv2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view), [VE-LOL](https://flyywh.github.io/IJCV2021LowLight_VELOL/) to prepare your data. 

If you want to test only, you should list your dataset as the followed rule:
```bash
    dataset/
        your_dataset/
            train/
                high/
                low/
            eval/
                high/
                low/
```

## 3. Pretrain Weights
The pretrain weight is at [Baidu Drive](https://pan.baidu.com/s/1EUGgDSsXebBygtw0yniTMA) (code: LD07). Please place it in the 'experience' directory of the 'DTCWT', 'HDR' and 'LDA' folders in the folder.

## 4. Testing
For low-light image enhancement testing, you need to modify the data path in the `config/gdrdnet_val.json` file, and then you can use the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py
```

Please note that recently many methods use the mean of the ground truth (GT-means) during testing, which may lead to better results. If you do not want to use this, you can disable it in test_from_dataset.py. We recommend ensuring consistent settings when making comparisons. (The results in `results` do not use GT-means.)

If the dataset is unpaired, please keep at least the same number and size of images in the `high` folder. It is normal for diffusion models to have some randomness, resulting in different enhancement outcomes.

**We found differences in the test results between machines equipped with RTX 3090 and RTX 4090. We recommend using only the RTX 3090 for testing to achieve the original performance. If you want to verify whether your results reflect the original performance, you can refer to the `results`.**

## 5. Train
If you need to train, you should train DTCWT, HDR, and LDA step by step. The training code is integrated into the `model` folder.

### Training the Encoder
You need to modify the dataset path for training in `model/DTCWT/train_decom.py`, and then you can use the following command:
```shell
# For DTCWT Frequency decomposition
python train_decom.py
```

### Training the HDR
Based on the weights trained for DTCWT, decompose the images from both the training set and the validation set, and organize the corresponding high-frequency maps dataset as follows:
```bash
    dataset/
        HDR_data/
            train/
                high/
                low/
            eval/
                high/
                low/
```
You need to modify the dataset path for training in `model/HDR/config` of HDR, and then you can use the following command:
```shell
# For HDR training
python train_hdr.py
```

### Training the LDA
Based on the weights trained for DTCWT, decompose the images from both the training set and the validation set, and organize the corresponding low-frequency maps dataset as follows:
(Additionally, you need to place the ground truth images of normal light in the `gt/` folder and the low-frequency maps restored from low light using HDR in the folder.)
```bash
    dataset/
        LDA_data/
            train/
                gt/
                R/
                high/
                low/
            eval/
                gt/
                R/
                high/
                low/
```
You need to modify the dataset path for training in `model/LDA/config` of LDA, and then you can use the following command:
```shell
# For LDA training
python train_lda.py
```
 
