import os
import mxnet as mx
import cv2
from tqdm import tqdm
import numpy as np
from imdbwiki_utils import get_meta

metafile='imdb'  #imdb or wiki
datapath='../../datasets/'
saveprefix = '../../datasets/imdb'
min_score=1.0

def main():
    id_x=0
    saveprefix_rec = saveprefix + '.rec'
    saveprefix_idx = saveprefix + '.idx'
    record = mx.recordio.MXIndexedRecordIO(saveprefix_idx,
                                           saveprefix_rec, 'w')
    full_path, dob, gender, photo_taken, face_score, second_face_score, age=get_meta(os.path.join(datapath,'%s.mat'%metafile),metafile)
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        fname=os.path.join(datapath,full_path)
        frame=cv2.imread(fname)
        if frame is None:
            continue
        nimg=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        header=mx.recordio.IRHeader(0,[age[i],gender[i]],0,0)
        s=mx.recordio.pack_img(header,nimg)
        record.write_idx(id_x,s)
        id_x+=1
        #print(id_x)
if __name__=='__main__':
    main()