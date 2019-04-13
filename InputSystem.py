#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/6 18:32
# @Author  : duwenzhi
# @Site    : 
# @File    : InputSystem.py
# @Software: PyCharm

import sys
import cv2
import os
import time
import tensorflow as tf
import mysql.connector
import numpy as np
import facenet
from scipy.misc import imresize
from align import detect_face
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow
from UI_Main import Ui_MainWindow

flags = tf.app.flags
root_path = os.getcwd() + os.sep
#初始化命令行参数
flags.DEFINE_string('img_dir', os.path.join(root_path + '', 'img/'), '图片保存目录')
flags.DEFINE_string('file_name', os.path.join(root_path + 'logs', 'data.txt'), '图片与中文姓名关联信息文件名称')
flags.DEFINE_string('model_path', os.path.join(root_path + '', '20170512-110547'), '预训练模型位置')

#初始化mysql数据库参数
config = {
        'user': 'root',
        'password': 'root',
        'host': '127.0.0.1',
        'database': 'test',
        'charset': 'utf8',
        'pool_size': 10,
        "pool_name": "server",
        "pool_reset_session": False
}

class PyQtMainEntry(QMainWindow, Ui_MainWindow):

    def __init__(self,FLAGS,mydb,is_save_mysql):
        super().__init__()
        with tf.Graph().as_default():
            with tf.Session() as sess:
                #加载预训练模型
                facenet.load_model(FLAGS.model_path)
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                #创建MTCNN模型,初始化pnet,rnet,onet网络，为摄像头获取的图片进行人脸对齐做准备
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

        #初始化UI界面
        self.setupUi(self)
        #打开摄像头
        self.camera = cv2.VideoCapture(0)
        #判断摄像头是否打开
        self.is_camera_opened = False

        # 定时器：30ms捕获一帧
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)

        self.FLAGS = FLAGS
        self.mydb = mydb
        # 是否保存到数据库
        self.is_save_mysql = is_save_mysql

    def btnOpenCamera_Clicked(self):
        '''
        打开和关闭摄像头
        '''
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.btnOpenCamera.setText('关闭摄像头')
            self._timer.start()
        else:
            self.btnOpenCamera.setText('打开摄像头')
            self._timer.stop()

    def btnCapture_Clicked(self):
        '''
        捕获图片
        '''
        # 摄像头未打开，不执行任何操作
        if not self.is_camera_opened:
            return

        self.captured = self.frame
        # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
        self.save_captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)
        t = time.time()
        #创建时间戳，由于opencv对中文支持存在问题，所以保存的图片名称都是去当前时间戳，精确到毫秒
        local_time = int(round(t * 1000))
        img_name = str(local_time)
        #对摄像头拍摄的图片进行人脸对齐、裁剪并保存
        self.image_array_align_data(self.save_captured,img_name)
        if not self.name_list :
            self.name_list.append('张三')
        name = self.name_list[0]
        save_name_info = name + '=' + img_name
        #保存每张人脸图片和对应姓名的关联关系
        self.save_data(save_name_info)
        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        # 使用Qt显示摄像头拍摄的图片，Qt显示图片时，需要先转换成QImgage类型
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def image_array_align_data(self, image_arr,img_name, image_size=160, margin=32,detect_multiple_faces=True):
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        img = image_arr
        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                # 进行图片缩放 cv2.resize(img,(w,h))
                scaled = imresize(cropped, (image_size, image_size), interp='bilinear')
                #保存人脸对齐、裁剪后的人脸图片
                cv2.imwrite(self.FLAGS.img_dir + img_name + '.jpg',scaled)

    def save_data(self,info):
        if self.is_save_mysql:
            mycursor = self.mydb.cursor()
            sql = 'INSERT INTO my_facenet_table (n_name,n_img_name) VALUES (%s, %s)'
            split = str(info).split('=')
            val = (str(split[0]), str(split[1]))
            mycursor.execute(sql, val)
            self.mydb.commit()
            print(mycursor.rowcount, '记录插入成功。')
        else:
            with open(self.FLAGS.file_name, 'a+', encoding='utf8') as f:
                f.write(info + '\n')
            print(info)

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        循环获取图片
        '''
        ret, self.frame = self.camera.read()
        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    #创建mysqlDB
    mydb = mysql.connector.connect(**config)
    #创建FLAGS
    FLAGS = tf.app.flags.FLAGS
    #是否保存到数据库
    is_save_mysql = True
    #创建PyQt5对象
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry(FLAGS,mydb,is_save_mysql)
    window.show()
    sys.exit(app.exec_())
