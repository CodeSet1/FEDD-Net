# GDRD-Net: a generative dual-branch Retinex diffusion with low-light diagonal features latent enhancement
### [Paper]() | [Code](https://github.com/CodeSet1/GDRD-Net)

This code is provided solely for academic research purposes and is exclusively restricted from any commercial use.

## 1. Create Environment
- Create Conda Environment
```
conda env create -f environment.yaml
```
- Activate Conda Environment
```
conda activate gdrd_net_env
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
The pretrain weight is at [Baidu Drive](https://pan.baidu.com/s/1EUGgDSsXebBygtw0yniTMA) (code: LD07). Please place it in the 'experience' directory of the 'DTCWT', 'RDR' and 'IDA' folders in the folder.

## 4. Testing
For low-light image enhancement testing, you need to modify the data path in the `config/gdrdnet_val.json` file, and then you can use the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py
```

Please note that recently many methods use the mean of the ground truth (GT-means) during testing, which may lead to better results. If you do not want to use this, you can disable it in test_from_dataset.py. We recommend ensuring consistent settings when making comparisons. (The results in `results` do not use GT-means.)

If the dataset is unpaired, please keep at least the same number and size of images in the `high` folder. It is normal for diffusion models to have some randomness, resulting in different enhancement outcomes.

**We found differences in the test results between machines equipped with RTX 3090 and RTX 4090. We recommend using only the RTX 3090 for testing to achieve the original performance. If you want to verify whether your results reflect the original performance, you can refer to the `results`.**

## 5. Train
If you need to train, you should train DTCWT, RDR, and IDA step by step. The training code is integrated into the `model` folder.

### Training the TDN
You need to modify the dataset path for training in `model/DTCWT/train_decom.py`, and then you can use the following command:
```shell
# For TDN training
python train_decom.py
```

### Training the RDR
Based on the weights trained for DTCWT, decompose the images from both the training set and the validation set, and organize the corresponding reflection maps dataset as follows:
```bash
    dataset/
        RDR_data/
            train/
                high/
                low/
            eval/
                high/
                low/
```
You need to modify the dataset path for training in `model/RDR/config` of RDR, and then you can use the following command:
```shell
# For RDR training
python train_rdr.py
```

### Training the IDA
Based on the weights trained for DTCWT, decompose the images from both the training set and the validation set, and organize the corresponding illumination maps dataset as follows:
(Additionally, you need to place the ground truth images of normal light in the `gt/` folder and the reflectance maps restored from low light using RDR in the `R/` folder.)
```bash
    dataset/
        IDA_data/
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
You need to modify the dataset path for training in `model/IDA/config` of IDA, and then you can use the following command:
```shell
# For IDA training
python train_ida.py
```
 
