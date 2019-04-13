#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/10 16:11
# @Author  : duwenzhi
# @Site    : 
# @File    : AttendanceSystem.py
# @Software: PyCharm

import os
import cv2
import facenet
import time
import datetime
import logging
import numpy as np
import mysql.connector
import tensorflow as tf
import align.detect_face
from scipy import misc
from PIL import Image,ImageDraw,ImageFont

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

flags = tf.app.flags
root_path = os.getcwd() + os.sep
logger.info('root_path : ' +root_path)
#初始化命令行参数
flags.DEFINE_string('model_path', os.path.join(root_path + '', '20170512-110547'), '预训练模型位置')
flags.DEFINE_string('img_dir', os.path.join(root_path + '', 'img'), '图片保存目录')
flags.DEFINE_string('file_name', os.path.join(root_path + 'logs', 'data.txt'), '图片与中文姓名关联信息文件名称')
flags.DEFINE_string('ttf_file', 'C:\WINDOWS\Fonts\simfang.ttf', '中文字体文件')
flags.DEFINE_integer('font_size',20,'中文字体大小')
flags.DEFINE_string('query_sql', 'SELECT * FROM my_facenet_table', '查询sql')

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

class PyFacenetDetect(object):

    def __init__(self,mydb,FLAGS,is_read_mysql):
        self.FLAGS = FLAGS
        if is_read_mysql:
            #获取保存的人脸图片与相对应的关联信息
            self.name_dict = self.read_mysql(mydb)
            self.mydb = mydb
        else:
            self.name_dict = self.read_log()
        #创建MTCNN模型,初始化pnet,rnet,onet网络，为摄像头获取的图片进行人脸对齐做准备
        pnet, rnet, onet = self.pre_net()
        #对摄像头拍摄的图片与录入的人脸图片进行识别
        self.detect(pnet, rnet, onet)


    def pre_net(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
        return pnet,rnet,onet

    def detect(self,pnet, rnet, onet):
        # tf.reset_default_graph()
        with tf.Session() as sess:
            # 加载预训练模型
            facenet.load_model(FLAGS.model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            ##录入的所有人脸图片
            image = []
            #录入的所有人脸图片名称
            all_img_list = []
            for i in os.listdir(FLAGS.img_dir):
                all_img_list.append(i)
                img = misc.imread(os.path.join(FLAGS.img_dir, i), mode='RGB')
                prewhitened = facenet.prewhiten(img)
                image.append(prewhitened)

            images = np.stack(image)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            #获取录入的所有人脸图片的128维向量
            compare_emb = sess.run(embeddings, feed_dict=feed_dict)
            compare_num = len(compare_emb)

            #打开摄像头
            capture = cv2.VideoCapture(0)
            cv2.namedWindow("camera", 1)
            while True:
                #获取摄像头的图片
                ret, frame = capture.read()
                if not ret:
                    break
                # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_PIL = Image.fromarray(rgb_frame)
                #使用ImageFont模块，显示中文字体
                font = ImageFont.truetype(FLAGS.ttf_file, FLAGS.font_size)
                # 获取摄像头拍摄的图片中人脸的标识以及对图片进行对齐、裁剪的人脸图片
                mark, bounding_box, crop_image = self.load_and_align_data(rgb_frame, 160, 44,pnet, rnet, onet)
                #判断图片中是否存在人脸
                if (mark):
                    feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
                    #获取对齐、裁剪后的人脸图片的128维向量
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    temp_num = len(emb)
                    output_text = []
                    #计算对齐、裁剪后的人脸图片的128维向量和所有录入的人脸信息的向量之间的距离
                    # for i in range(temp_num):
                    dist_list = []
                    for j in range(compare_num):
                        #计算向量之间的距离
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], compare_emb[j, :]))))
                        dist_list.append(dist)
                    min_value = min(dist_list)
                    if (min_value > 0.65):
                        output_text.append('无法识别,请重新录入信息')
                    else:
                        output_text.append(self.name_dict[all_img_list[dist_list.index(min_value)].split('.jpg')[0]] + '考勤成功')
                        self.record_attendance_info(self.name_dict[all_img_list[dist_list.index(min_value)].split('.jpg')[0]])


                    # 在frame上绘制边框和文字
                    # for rec_position in range(temp_num):
                    # 字体颜色
                    fillColor = (255, 255, 255)
                    # 文字输出位置
                    position = (200, 400)
                    # 输出内容
                    str = output_text[0]
                    draw = ImageDraw.Draw(img_PIL)
                    draw.text(position, str, font=font, fill=fillColor)

                    # 转换回OpenCV格式
                    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                    cv2.resizeWindow('camera', 550, 450)
                    cv2.imshow('camera', img_OpenCV)

                # time.sleep(2.0)
                key = cv2.waitKey(3)
                if key == 27:
                    break
            capture.release()
            cv2.destroyWindow('camera')

    def record_attendance_info(self,name):
        mycursor = mydb.cursor()
        local_times = datetime.datetime.now().strftime('%Y-%m-%d')
        query_time_sql = "select * from my_attendance_table where n_name =%s and n_update_time like %s order by n_id;"
        val = (name, str(local_times) + "%")
        mycursor.execute(query_time_sql, val)
        myresult = mycursor.fetchall()
        if myresult:
            if len(myresult) == 2:
                id, name, time = myresult[1]
                local_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                update_sql = "UPDATE my_attendance_table SET n_name = %s,n_update_time=%s WHERE n_id = %s"
                val = (name, local_timestamp, id)
                mycursor.execute(update_sql, val)
                mydb.commit()
                logger.info(str(mycursor.rowcount)+" 条记录被修改")
            else:
                sql = "INSERT INTO my_attendance_table (n_name) VALUES (%s)"
                val = [name]
                mycursor.execute(sql, val)
                mydb.commit()
                logger.info(str(mycursor.rowcount)+" 记录插入成功")
        else:
            sql = "INSERT INTO my_attendance_table (n_name) VALUES (%s)"
            val = [name]
            mycursor.execute(sql, val)
            mydb.commit()
            logger.info(str(mycursor.rowcount)+" 记录插入成功")


    def load_and_align_data(self,img, image_size, margin,pnet, rnet, onet):
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        if len(bounding_boxes) < 1:
            return 0, 0, 0
        det = bounding_boxes
        det[:, 0] = np.maximum(det[:, 0] - margin / 2, 0)
        det[:, 1] = np.maximum(det[:, 1] - margin / 2, 0)
        det[:, 2] = np.minimum(det[:, 2] + margin / 2, img_size[1] - 1)
        det[:, 3] = np.minimum(det[:, 3] + margin / 2, img_size[0] - 1)

        det = det.astype(int)
        crop = []
        for i in range(len(bounding_boxes)):
            temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
            aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)

        crop_image = np.stack(crop)
        return 1, det, crop_image

    def read_mysql(self,mydb):
        mycursor = mydb.cursor()
        mycursor.execute(self.FLAGS.query_sql)
        myresult = mycursor.fetchall()
        name_dict = {}
        for x in myresult:
            id, name, img_id = x
            name_dict[img_id] = name
        return name_dict

    def read_log(self):
        name_dict = {}
        print(self.FLAGS.file_name)
        with open(self.FLAGS.file_name,'r',encoding='utf8') as f:
            for line in f.readlines():
                line.strip('\n').split('=')[0]
                name_dict[line.strip('\n').split('=')[1]] = line.strip('\n').split('=')[0]
        return name_dict

def init_mysqldb():
    try:
        mydb = mysql.connector.connect(**config)
    except mysql.connector.Error as e:
        logger.error('connect fails!{}'.format(e))
    logger.info('connect create success!')
    return mydb

if __name__=='__main__':
    #按ESC键，退出
    FLAGS = tf.app.flags.FLAGS
    mydb = init_mysqldb()
    # 是否读取数据库
    is_read_mysql = True
    facenet_detect = PyFacenetDetect(mydb,FLAGS,is_read_mysql)
