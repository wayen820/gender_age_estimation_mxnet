import mxnet as mx
import cv2
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'align'))
from align_tools import align_tools

'''
this file is used for test slim model
'''
model_str='../../models/ssr2_megaage_1_1/model,0'
model_gender_str='../../models/ssr2_imdb_gender_1_1/model,0'
gpu=0

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
  model.set_params(arg_params, aux_params)
  return model

def get_model_gender(ctx, image_size, model_str, layer):
  _vec = model_gender_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
  model.set_params(arg_params, aux_params)
  return model

def main():
    align_t=align_tools()
    cap=cv2.VideoCapture(0)
    model=get_model(mx.gpu(gpu),(64,64),model_str,'_mulscalar16')
    model_gender=get_model_gender(mx.gpu(gpu),(64,64),model_gender_str,'_mulscalar16')
    while(1):
        ret,frame=cap.read()
        if ret is False:
            continue
        nimg,box=align_t.get_intput_cv(frame)
        if nimg is None:
            continue
        nimg = nimg[:, :, ::-1]
        nimg= np.transpose(nimg,(2,0,1))


        input_blob = np.expand_dims(nimg, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))
        model.forward(db, is_train=False)
        age = model.get_outputs()[0].asnumpy()

        model_gender.forward(db,is_train=False)
        gender=model_gender.get_outputs()[0].asnumpy()
        # print('%s,%s'%(gender,age))
        # print(box)
        g='female'
        if gender[0]>0.5:
            g='male'
        cv2.putText(frame,'%s-%.2f'%(g,age[0]),(int(box[0]),int(box[1])),1,3,(0,0,255),3)
        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),5)
        cv2.imshow('capture',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
