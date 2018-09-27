import os
import sys
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import mxnet as mx
import cv2
import numpy as np


class align_tools:
    def __init__(self,det=0,image_size=(64,64)):
        '''
        人脸对齐工具
        :param det:0全部检测，1关键点检测和对齐，用于已经crop的人脸图像
        :param image_size:
        '''
        self.det=det
        self.image_size=image_size
        self.ctx=mx.gpu(0)
        det_threshold = [0.6, 0.7, 0.8]
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        if det == 0:
            self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True,
                                     threshold=det_threshold)
        else:
            self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
    def get_intput_cv(self,face_img):
        '''
        返回对齐的图像，格式HWC,bgr，如果图像有多张人脸，优先选择位于图像中间位置同时大小比较大的人脸
        :param img_file: 图像路径
        :return: align face image，format HWC bgr
        '''
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None,None
        bounding_boxes, points = ret
        if bounding_boxes.shape[0] == 0:
            return None,None
        nrof_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(face_img.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        _bbox = bounding_boxes[bindex, 0:4]
        _landmark = points[bindex, :].reshape((2, 5)).T
        warped = face_preprocess.preprocess(face_img, bbox=_bbox, landmark=_landmark)
        warped = cv2.resize(warped, self.image_size)
        return warped,_bbox

    def get_input(self, img_file):
        '''
        返回对齐的图像，格式HWC,bgr，如果图像有多张人脸，优先选择位于图像中间位置同时大小比较大的人脸
        :param img_file: 图像路径
        :return: align face image，format HWC bgr
        '''
        face_img=cv2.imread(img_file)
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None
        bounding_boxes, points = ret
        if bounding_boxes.shape[0] == 0:
            return None
        nrof_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(face_img.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        _bbox = bounding_boxes[bindex, 0:4]
        _landmark = points[bindex,:].reshape((2, 5)).T
        warped = face_preprocess.preprocess(face_img, bbox=_bbox, landmark=_landmark)
        warped= cv2.resize(warped,self.image_size)
        return warped




