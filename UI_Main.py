#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/6 18:33
# @Author  : duwenzhi
# @Site    :
# @File    : UI.py
# @Software: PyCharm

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (QPushButton, QLabel, QInputDialog)


class Ui_MainWindow(object):

    def __init__(self):
        self.name_list = []

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.setGeometry(50, 50, 1123, 662)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnOpenCamera = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenCamera.setGeometry(QtCore.QRect(620, 540, 93, 28))
        self.btnOpenCamera.setObjectName("btnOpenCamera")
        self.labelCamera = QtWidgets.QLabel(self.centralwidget)
        self.labelCamera.setGeometry(QtCore.QRect(10, 10, 550, 450))
        self.labelCamera.setObjectName("labelCamera")
        self.btnCapture = QtWidgets.QPushButton(self.centralwidget)
        self.btnCapture.setGeometry(QtCore.QRect(920, 540, 93, 28))
        self.btnCapture.setObjectName("btnCapture")
        self.labelCapture = QtWidgets.QLabel(self.centralwidget)
        self.labelCapture.setGeometry(QtCore.QRect(565, 10, 550, 450))
        self.labelCapture.setObjectName("labelCapture")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1123, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.btn = QPushButton('请输入姓名', self)
        self.btn.move(20, 540)

        self.le = QLabel("张三",self)
        self.le.move(320, 540)

        self.retranslateUi(MainWindow)
        self.btnOpenCamera.clicked.connect(MainWindow.btnOpenCamera_Clicked)
        self.btnCapture.clicked.connect(MainWindow.btnCapture_Clicked)
        self.btn.clicked.connect(self.showDialog)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "人脸识别考勤录入界面"))
        self.btnOpenCamera.setText(_translate("MainWindow", "打开摄像头"))
        self.btnCapture.setText(_translate("MainWindow", "拍照"))

    def showDialog(self):
        text, ok = QInputDialog.getText(self, '输入框',
                                        '请输入你的姓名:')
        if ok:
            self.le.setText(str(text))
            if self.name_list:
                self.name_list.pop()
            self.name_list.append(str(text))