# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\hw1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 1. Image Processing
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 20, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")

        # 2. Image Smoothing
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(223, 20, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        # 3. Image Transformation
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(565, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        # 4. Edge Detection
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(392, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        # 5. Training Cifar10 Classifier Using VGG16
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(812, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")  

        # layout of 1
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(50, 50, 160, 240))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")

        # 1.1 Load Image
        self.btn1_1 = QtWidgets.QPushButton(self.frame)
        self.btn1_1.setGeometry(QtCore.QRect(10, 10, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_1.setFont(font)
        self.btn1_1.setObjectName("btn1_1")

        # 1.2 Color Separation
        self.btn1_2 = QtWidgets.QPushButton(self.frame)
        self.btn1_2.setGeometry(QtCore.QRect(10, 70, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_2.setFont(font)
        self.btn1_2.setObjectName("btn1_2")

        # 1.3 Image Flipping
        self.btn1_3 = QtWidgets.QPushButton(self.frame)
        self.btn1_3.setGeometry(QtCore.QRect(10, 130, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_3.setFont(font)
        self.btn1_3.setObjectName("btn1_3")

        # 1.4 Blending
        self.btn1_4 = QtWidgets.QPushButton(self.frame)
        self.btn1_4.setGeometry(QtCore.QRect(10, 190, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_4.setFont(font)
        self.btn1_4.setObjectName("btn1_4")

       # layout of 2
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(220, 50, 160, 240))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName("frame_2")

        # 2.1 Median Filter
        self.btn2_1 = QtWidgets.QPushButton(self.frame_2)
        self.btn2_1.setGeometry(QtCore.QRect(10, 40, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_1.setFont(font)
        self.btn2_1.setObjectName("btn2_1")
        
        # 2.2 Gaussian Blur
        self.btn2_2 = QtWidgets.QPushButton(self.frame_2)
        self.btn2_2.setGeometry(QtCore.QRect(10, 100, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_2.setFont(font)
        self.btn2_2.setObjectName("btn2_2")

        # 2.3 Bilateral filter
        self.btn2_3 = QtWidgets.QPushButton(self.frame_2)
        self.btn2_3.setGeometry(QtCore.QRect(10, 160, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_3.setFont(font)
        self.btn2_3.setObjectName("btn2_3")

        # layout of 4
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(560, 50, 240, 240))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_4.setObjectName("frame_4")

        # 4.1 button
        self.btn4_1 = QtWidgets.QPushButton(self.frame_4)
        self.btn4_1.setGeometry(QtCore.QRect(15, 180, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_1.setFont(font)
        self.btn4_1.setObjectName("btn4_1")
 
        self.label_4_R = QtWidgets.QLabel(self.frame_4)
        self.label_4_R.setGeometry(QtCore.QRect(10, 20, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4_R.setFont(font)
        self.label_4_R.setObjectName("label_4_R")

        self.label_4_S = QtWidgets.QLabel(self.frame_4)
        self.label_4_S.setGeometry(QtCore.QRect(10, 60, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4_S.setFont(font)
        self.label_4_S.setObjectName("label_4_S")

        self.label_4_Tx = QtWidgets.QLabel(self.frame_4)
        self.label_4_Tx.setGeometry(QtCore.QRect(10, 100, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4_Tx.setFont(font)
        self.label_4_Tx.setObjectName("label_4_Tx")

        self.label_4_Ty = QtWidgets.QLabel(self.frame_4)
        self.label_4_Ty.setGeometry(QtCore.QRect(10, 140, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4_Ty.setFont(font)
        self.label_4_Ty.setObjectName("label_4_Ty")

        self.edtAngle = QtWidgets.QLineEdit(self.frame_4)
        self.edtAngle.setGeometry(QtCore.QRect(60, 20, 113, 20))
        self.edtAngle.setObjectName("edtAngle")

        self.edtScale = QtWidgets.QLineEdit(self.frame_4)
        self.edtScale.setGeometry(QtCore.QRect(60, 60, 113, 20))
        self.edtScale.setObjectName("edtScale")
        self.edtTx = QtWidgets.QLineEdit(self.frame_4)
        self.edtTx.setGeometry(QtCore.QRect(60, 100, 113, 20))
        self.edtTx.setObjectName("edtTx")

        self.edtTy = QtWidgets.QLineEdit(self.frame_4)
        self.edtTy.setGeometry(QtCore.QRect(60, 140, 113, 20))
        self.edtTy.setObjectName("edtTy")

        self.label_4_deg = QtWidgets.QLabel(self.frame_4)
        self.label_4_deg.setGeometry(QtCore.QRect(180, 20, 21, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_4_deg.setFont(font)
        self.label_4_deg.setObjectName("label_4_deg")

        self.label_4_pix1 = QtWidgets.QLabel(self.frame_4)
        self.label_4_pix1.setGeometry(QtCore.QRect(180, 100, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_4_pix1.setFont(font)
        self.label_4_pix1.setObjectName("label_4_pix1")

        self.label_4_pix2 = QtWidgets.QLabel(self.frame_4)
        self.label_4_pix2.setGeometry(QtCore.QRect(180, 140, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_4_pix2.setFont(font)
        self.label_4_pix2.setObjectName("label_4_pix2")


        # layout of 3
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(390, 50, 160, 240))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setObjectName("frame_3")

        self.btn3_1 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_1.setGeometry(QtCore.QRect(10, 10, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_1.setFont(font)
        self.btn3_1.setObjectName("btn3_1")

        self.btn3_2 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_2.setGeometry(QtCore.QRect(10, 70, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_2.setFont(font)
        self.btn3_2.setObjectName("btn3_2")

        self.btn3_3 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_3.setGeometry(QtCore.QRect(10, 130, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_3.setFont(font)
        self.btn3_3.setObjectName("btn3_3")

        self.btn3_4 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_4.setGeometry(QtCore.QRect(10, 190, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_4.setFont(font)
        self.btn3_4.setObjectName("btn3_4")

        # layout of 5
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(810, 50, 271, 300))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_5.setObjectName("frame_5")

        # layout of button5.1
        self.btn5_1 = QtWidgets.QPushButton(self.frame_5)
        self.btn5_1.setGeometry(QtCore.QRect(25, 5, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_1.setFont(font)
        self.btn5_1.setObjectName("btn5_1")

        # layout of button5.2
        self.btn5_2 = QtWidgets.QPushButton(self.frame_5)
        self.btn5_2.setGeometry(QtCore.QRect(25, 50, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_2.setFont(font)
        self.btn5_2.setObjectName("btn5_2")

        # layout of button5.3
        self.btn5_3 = QtWidgets.QPushButton(self.frame_5)
        self.btn5_3.setGeometry(QtCore.QRect(25, 95, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_3.setFont(font)
        self.btn5_3.setObjectName("btn5_3")

        # layout of button5.4
        self.btn5_4 = QtWidgets.QPushButton(self.frame_5)
        self.btn5_4.setGeometry(QtCore.QRect(25, 140, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_4.setFont(font)
        self.btn5_4.setObjectName("btn5_4")

        # layout of button5.5
        self.btn5_5 = QtWidgets.QPushButton(self.frame_5)
        self.btn5_5.setGeometry(QtCore.QRect(25, 225, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_5.setFont(font)
        self.btn5_5.setObjectName("btn5_5")

        # layout of SpinBox
        self.SpinBox5 = QtWidgets.QSpinBox(self.frame_5)
        self.SpinBox5.setGeometry(QtCore.QRect(25, 185, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.SpinBox5.setFont(font)
        self.SpinBox5.setRange(0, 9999)
        self.SpinBox5.setObjectName("SpinBox5")   

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 682, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Median Filter"))
        self.btn2_2.setText(_translate("MainWindow", "2.2 Gaussian Blur"))
        self.btn2_3.setText(_translate("MainWindow", "2.3 Bilateral filter"))

        self.btn3_1.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.btn3_2.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.btn3_3.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.btn3_4.setText(_translate("MainWindow", "3.4 Magnitude"))
        
        self.label.setText(_translate("MainWindow", "1. Image Processing"))
        self.label_2.setText(_translate("MainWindow", "2. Image Smoothing"))
        self.label_3.setText(_translate("MainWindow", "3. Edge Detection"))
        self.label_4.setText(_translate("MainWindow", "4. Image Transformation"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Load Image"))        
        self.btn1_2.setText(_translate("MainWindow", "1.2 Color Separation"))
        self.btn1_3.setText(_translate("MainWindow", "1.3 Image Flipping"))
        self.btn1_4.setText(_translate("MainWindow", "1.4 Blending"))
        self.btn4_1.setText(_translate(
            "MainWindow", "4.1 Rotation, Scaling, Translation"))

        self.label_4_R.setText(_translate("MainWindow", "Rotation:"))
        self.label_4_S.setText(_translate("MainWindow", "Scaling:"))
        self.label_4_Tx.setText(_translate("MainWindow", "Tx:"))
        self.label_4_Ty.setText(_translate("MainWindow", "Ty:"))
        self.label_4_deg.setText(_translate("MainWindow", "deg"))
        self.label_4_pix1.setText(_translate("MainWindow", "pixel"))
        self.label_4_pix2.setText(_translate("MainWindow", "pixel"))
        self.label_5.setText(_translate("MainWindow", "5. Cifar10 Classifier"))        
        self.btn5_1.setText(_translate("MainWindow", "5.1 Show Train Image"))
        self.btn5_2.setText(_translate("MainWindow", "5.2 Show hyperparameters"))
        self.btn5_3.setText(_translate("MainWindow", "5.3 Show Model Structure"))
        self.btn5_4.setText(_translate("MainWindow", "5.4 Show Accuracy"))
        self.btn5_5.setText(_translate("MainWindow", "5.5 Test"))