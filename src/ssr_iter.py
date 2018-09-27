import mxnet as mx
import logging
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
import random
import numpy as np
import cv2
# pylint: disable=g-import-not-at-top
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
try:
    from scipy import linalg
    import scipy.ndimage as ndi
except ImportError:
    linalg = None
    ndi = None
from moviepy.editor import *


# pylint: enable=g-import-not-at-top


# logger = logging.getLogger()

class SSR_ITER(mx.io.DataIter):
    def __init__(self,batch_size,data_shape,path_imgrec=None,shuffle=False,data_name='data',gender_label_name='label_gender',
                 age_lable_name='label_age',**kwargs):
        super(SSR_ITER,self).__init__()
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
        # self.rand_mirror = rand_mirror
        # print('rand_mirror', rand_mirror)
        #self.provide_label=[(gender_label_name,(batch_size,1)),(age_lable_name, (batch_size,1))]
        self.provide_label = [(age_lable_name, (batch_size,))]
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
        # gender_label=nd.empty((batch_size,self.provide_label[0][1]))
        age_label=nd.empty(self.provide_label[0][1])
        i=0

        try:
            while i<batch_size:
                label, s, _, _=self.next_sample()
                _data=mx.image.imdecode(s).asnumpy() #imdecode输入为opencv格式，输出会将bgr=>rgb
                # if True:
                #     cv2.imshow('_data',_data[:,:,::-1])
                #     key = cv2.waitKey(1000)

                if np.random.random()>0.5:
                    _data=_data[:,::-1]
                if np.random.random()>0.75:
                    _data=self.random_rotation(x=_data,rg=20,row_axis=0,col_axis=1,channel_axis=2)
                if np.random.random()>0.75:
                    _data=self.random_shear(x=_data,intensity=0.2,row_axis=0,col_axis=1,channel_axis=2)
                if np.random.random()>0.75:
                    _data=self.random_shift(x=_data,wrg=0.2,hrg=0.2,row_axis=0,col_axis=1,channel_axis=2)
                if np.random.random()>0.75:
                    _data=self.random_zoom(x=_data,zoom_range=[0.8,1.2],row_axis=0,col_axis=1,channel_axis=2)

                batch_data[i][:]=self.postprocess_data(mx.nd.array(_data))
                # gender_label[i][:]=label[1]
                age_label[i][:]=label[0]
                i+=1
                # if True:
                #     cv2.imshow('aur_data',_data[:,:,::-1])
                #     cv2.waitKey(1000)
        except StopIteration:
            if i<batch_size:
                raise StopIteration
        return io.DataBatch([batch_data], [age_label], batch_size - i)

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

    def random_rotation(self,x,
                        rg,
                        row_axis=1,
                        col_axis=2,
                        channel_axis=0,
                        fill_mode='nearest',
                        cval=0.):
        """Performs a random rotation of a Numpy image tensor.
        Arguments:
            x: Input tensor. Must be 3D.
            rg: Rotation range, in degrees.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        Returns:
            Rotated Numpy image tensor.
        """
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self.transform_matrix_offset_center(rotation_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_shift(self,x,
                     wrg,
                     hrg,
                     row_axis=1,
                     col_axis=2,
                     channel_axis=0,
                     fill_mode='nearest',
                     cval=0.):
        """Performs a random spatial shift of a Numpy image tensor.
        Arguments:
            x: Input tensor. Must be 3D.
            wrg: Width shift range, as a float fraction of the width.
            hrg: Height shift range, as a float fraction of the height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        Returns:
            Shifted Numpy image tensor.
        """
        h, w = x.shape[row_axis], x.shape[col_axis]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        transform_matrix = translation_matrix  # no need to do offset
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_shear(self,x,
                     intensity,
                     row_axis=1,
                     col_axis=2,
                     channel_axis=0,
                     fill_mode='nearest',
                     cval=0.):
        """Performs a random spatial shear of a Numpy image tensor.
        Arguments:
            x: Input tensor. Must be 3D.
            intensity: Transformation intensity.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        Returns:
            Sheared Numpy image tensor.
        """
        shear = np.random.uniform(-intensity, intensity)
        shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self.transform_matrix_offset_center(shear_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def random_zoom(self,x,
                    zoom_range,
                    row_axis=1,
                    col_axis=2,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
        """Performs a random spatial zoom of a Numpy image tensor.
        Arguments:
            x: Input tensor. Must be 3D.
            zoom_range: Tuple of floats; zoom range for width and height.
            row_axis: Index of axis for rows in the input tensor.
            col_axis: Index of axis for columns in the input tensor.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        Returns:
            Zoomed Numpy image tensor.
        Raises:
            ValueError: if `zoom_range` isn't a tuple.
        """
        if len(zoom_range) != 2:
            raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self.transform_matrix_offset_center(zoom_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
        return x

    def transform_matrix_offset_center(self,matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def apply_transform(self,x,
                        transform_matrix,
                        channel_axis=0,
                        fill_mode='nearest',
                        cval=0.):
        """Apply the image transformation specified by a matrix.
        Arguments:
            x: 2D numpy array, single image.
            transform_matrix: Numpy array specifying the geometric transformation.
            channel_axis: Index of axis for channels in the input tensor.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
        Returns:
            The transformed version of the input.
        """
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [
            ndi.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=0,
                mode=fill_mode,
                cval=cval) for x_channel in x
        ]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x