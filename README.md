# gender_age_estimation_mxnet
**[IJCAI18] SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation**
+ A real-time age estimation model with 170KB.
+ Gender regression is also added!
+ Megaage-Asian is provided in https://github.com/b02901145/SSR-Net_megaage-asian
+ Coreml model (0.17MB) is provided in https://github.com/shamangary/Keras-to-coreml-multiple-inputs-example

This is a mxnet version implementation of SSR-Net for age and gender Estimation,Keras version in here https://github.com/shamangary/SSR-Net,but we get better accuracy.

### Real-time webcam demo
<img src="https://github.com/wayen820/gender_age_estimation_mxnet/raw/master/test.gif" height="240"/>
## Paper

### PDF
https://github.com/wayen820/gender_age_estimation_mxnet/raw/master/ssr.pdf

### Paper authors
**[Tsun-Yi Yang](http://shamangary.logdown.com/), [Yi-Husan Huang](https://github.com/b02901145), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), [Pi-Cheng Hsiu](https://www.citi.sinica.edu.tw/pages/pchsiu/index_en.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

## Abstract
This paper presents a novel CNN model called Soft Stagewise Regression Network (SSR-Net) for age estimation from a single image with a compact model size. Inspired by DEX, we address age estimation by performing multi-class classification and then turning classification results into regression by calculating the expected values. SSR-Net takes a coarse-to-fine strategy and performs multi-class classification with multiple stages. Each stage is only responsible for refining the decision of the previous stage. Thus, each stage performs a task with few classes and requires few neurons, greatly reducing the model size. For addressing the quantization issue introduced by grouping ages into classes, SSR-Net assigns a dynamic range to each age class by allowing it to be shifted and scaled according to the input face image. Both the multi-stage strategy and the dynamic range are incorporated into the formulation of soft stagewise regression. A novel network architecture is proposed for carrying out soft stagewise regression. The resultant SSR-Net model is very compact and takes only **0.32 MB**. Despite of its compact size, SSR-Net’s performance approaches those of the state-of-the-art methods whose model sizes are more than 1500x larger.

## Platform
+ Mxnet
+ Tensorflow
+ GTX-1080Ti
+ Ubuntu

## Dependencies

## Codes

There are three different section of this project.
1. Data pre-processing
2. Training
3. Model modification and Test
We will go through the details in the following sections.

This repository is for IMDB, WIKI, and Megaage datasets.


### 1. Data pre-processing
+ You can download rec format file directly in here: https://share.weiyun.com/5rQwJtA ,or package it youself according to the following steps
+ Download IMDB-WIKI dataset (face only) from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.
+ Download Megaage dataset from https://github.com/b02901145/SSR-Net_megaage-asian
+ Unzip them under '~/data' or others path
+ Run the following codes for dataset pre-processing.
```
cd ./src/data
python3 ./process_data_wiki_imdb.py --rootpath ~/data/imdb_crop --outputpath ../../datasets/imdb
python3 ./process_data_wiki_imdb.py --rootpath ~/data/wiki_crop --outputpath ../../datasets/wiki
python3 ./process_data_mege_asia.py --rootpath ~/data/megaage/megaage_asian/train --agefile ~/data/megaage/megaage_asian/list/train_age.txt --namefile ~/data/megaage/megaage_asian/list/train_name.txt --saveprefix ../../datasets/megaage/train
python3 ./process_data_mege_asia.py --rootpath ~/data/megaage/megaage_asian/test --agefile ~/data/megaage/megaage_asian/list/test_age.txt --namefile ~/data/megaage/megaage_asian/list/test_name.txt --saveprefix ../../datasets/megaage/test
```

### 2. Training
For Age Train,first train on imdb ,then fine tune on megaage-asia
```
cd ./src
./train_ssr_adam.sh
```
For gender Train,train on imdb directly
```
cd ./src
./train_ssr_adam_gender.sh
```
### 3. Model modification and Test
Some inference frameworks like ncnn do not support arange ops,but we need stage_num parameter,so we use them as input. if you do not want deploy,you don't need do this.
model modification please reference ./src/deploy/model_slim_gender.py and ./src/deploy/model_slim_age.py.I have provided pre-trained model in models directory,you can use it directly.

Test from web camera:
```
cd ./src/deploy/
python3 ./test.py
```

