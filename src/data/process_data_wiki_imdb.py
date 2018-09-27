import os
import mxnet as mx
import cv2
from tqdm import tqdm
import numpy as np
from imdbwiki_utils import get_meta
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'align'))
from align_tools import align_tools
import random
#这个文件用来打包imdb或者wiki数据到rec格式

metafile='imdb'  #imdb or wiki
rootpath='/media/kneron/Data/datasets/wiki_imdb/imdb_crop'  #imdb or wiki 的图片路径
outputpath='../../datasets/imdb_raw/'
# saveprefix_train = '../../datasets/train'  #数据保存路径前缀
# saveprefix_val = '../../datasets/val'  #数据保存路径前缀
ratio=0.8 #训练集占比
min_score=1.0
align=False  #是否执行检测人脸和对齐


def main():
    align_t=align_tools()
    id_x=0
    id_x1=0
    nok=0
    total=0
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    saveprefix_rec = outputpath+'train' + '.rec'
    saveprefix_idx = outputpath+'train' + '.idx'
    record = mx.recordio.MXIndexedRecordIO(saveprefix_idx,
                                           saveprefix_rec, 'w')

    saveprefix_rec1 = outputpath+'val' + '.rec'
    saveprefix_idx1 = outputpath+'val' + '.idx'
    record1 = mx.recordio.MXIndexedRecordIO(saveprefix_idx1,
                                           saveprefix_rec1, 'w')

    full_path, dob, gender, photo_taken, face_score, second_face_score, age=get_meta(os.path.join(rootpath,'%s.mat'%metafile),metafile)

    # train_count=len(face_score)*ratio
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue
        fname=os.path.join(rootpath,str(full_path[i][0]))
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
        header = mx.recordio.IRHeader(0, [age[i], gender[i]], 0, 0)
        s = mx.recordio.pack_img(header, nimg)
        if random.random()<ratio:
            record.write_idx(id_x,s)
            id_x+=1
        else:
            record1.write_idx(id_x1,s)
            id_x1+=1
    print('total:%d,nok:%d'%(total,nok))
if __name__=='__main__':
    main()