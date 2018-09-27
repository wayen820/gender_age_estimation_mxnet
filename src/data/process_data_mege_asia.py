import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
from moviepy.editor import *
import mxnet as mx
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'align'))
from align_tools import align_tools
import cv2
import os

#这个文件用来打包megaage数据到rec格式

metafile='wiki'  #imdb or wiki
listfile='/media/kneron/Data/datasets/megaage/megaage_asian/list/train_age.txt'  #imdb or wiki 的图片路径
imgnamefile='/media/kneron/Data/datasets/megaage/megaage_asian/list/train_name.txt'
rootpath='/media/kneron/Data/datasets/megaage/megaage_asian/train'
saveprefix = '../../datasets/megaage/train'  #数据保存路径前缀
min_score=1.0
align=False  #是否执行检测人脸和对齐


def main():
    align_t=align_tools()
    id_x=0
    nok=0
    total=0
    if not os.path.exists(os.path.dirname(saveprefix)):
        os.mkdir(os.path.dirname(saveprefix))
    saveprefix_rec = saveprefix + '.rec'
    saveprefix_idx = saveprefix + '.idx'
    record = mx.recordio.MXIndexedRecordIO(saveprefix_idx,
                                           saveprefix_rec, 'w')
    age_file = np.loadtxt(listfile)
    img_name_file = np.genfromtxt(imgnamefile, dtype='str')
    for i in tqdm(range(len(img_name_file))):
        fname=os.path.join(rootpath,str(img_name_file[i]))
        age = int(float(age_file[i]))
        if age >= -1:
            total+=1
            if  align:
                nimg=align_t.get_input(fname)
                if nimg is None:
                    nok+=1
                    continue
            else:
                frame = cv2.imread(fname)
                if frame is None:
                    continue
                nimg=cv2.resize(frame,(64,64))
            # cv2.imshow('nimg',nimg)
            # cv2.waitKey(1000)

            header = mx.recordio.IRHeader(0, [age,-1], 0, 0)
            s = mx.recordio.pack_img(header, nimg)
            record.write_idx(id_x,s)
            id_x+=1
    print('total:%d,nok:%d'%(total,nok))
if __name__ == '__main__':
    main()
