import mxnet as mx
import logging
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import random

logger = logging.getLogger()

class Gender_Age_Iter(mx.io.DataIter):
    def __init__(self,batch_size,data_shape,path_imgrec=None,shuffle=False,rand_mirror=False,data_name='data',gender_label_name='softmax_gender',
                 age_lable_name='softmax_age',**kwargs):
        super(Gender_Age_Iter,self).__init__()
        assert path_imgrec
        self.batch_size = self.batch_size
        logging.info('loading recordio %s...',path_imgrec)
        path_imgidx=path_imgrec[0:-4]+".idx"
        self.imgrec=recordio.MXIndexedRecordIO(path_imgidx,path_imgrec,'r')
        self.imgidx=list(self.imgrec.keys)
        if shuffle:
            self.seq=self.imgidx
            self.oseq=self.imgidx
        else:
            self.seq=None
        self.provide_data=[(data_name,(batch_size,)+data_shape)]
        self.batch_size=batch_size
        self.data_shape=data_shape
        self.shuffle=shuffle
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.provide_label=[(gender_label_name, 102),(age_lable_name, 2)]
        self.cur = 0
        self.nbatch = 0
        self.is_init = False

    def reset(self):
        print('call reset')
        self.cur=0
        if self.shuffle:
          random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    def num_samples(self):
        return len(self.seq)

    def next_sample(self):
        if self.seq is not None:
            while True:
                if self.cur>=len(self.seq):
                    raise StopIteration
                idx=self.seq[self.cur]
                self.cur+=1
                if self.imgrec is not None:
                    s=self.imgrec.read_idx(idx)
                    header,img=recordio.unpack(s)
                    return header.label,img,None,None
        else:
            s=self.imgrec.read()
            if s is None:
                raise StopIteration
            header,img=recordio.unpack(s)
            return header.label,img,None,None
    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init=True
        self.nbatch+=1
        batch_size=self.batch_size
        c,h,w=self.data_shape
        batch_data=nd.empty((batch_size,c,h,w))
        gender_label=nd.empty(self.provide_label[0][1])
        age_label=nd.empty(self.provide_label[1][1])
        i=0
        try:
            while i<batch_size:
                label, s, _, _=self.next_sample()
                _data=mx.image.imdecode(s)
                if self.rand_mirror:
                    _rd=random.randint(0,1)
                    if _rd==1:
                        _data=mx.ndarray.flip(data=_data,axis=1)
                if _data.shape[0]==0:
                    logging.debug('Invalid image,skipping')
                    continue
                batch_data[i][:]=self.postprocess_data(_data)
                gender_label[i][:]=label[0]
                age_label[i][:]=label[1]
                i+=1
        except StopIteration:
            if i<batch_size:
                raise StopIteration
        return io.DataBatch([batch_data], [gender_label,age_label], batch_size - i)

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))