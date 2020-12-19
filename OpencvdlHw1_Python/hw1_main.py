# -*- coding: utf-8 -*-
import os
import sys
import glob
import argparse
import cv2 as cv
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QMainWindow, QApplication
from hw1_ui import Ui_MainWindow
from scipy import signal
from scipy import misc
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import model_from_json
import keras.backend as K

model = load_model('final_model.h5')
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data()
cont3 = 0
row = 250
column = 461
sobelxImage = np.zeros((row,column))
sobelyImage = np.zeros((row,column))
sobelGrad = np.zeros((row,column))
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)
        self.btn5_3.clicked.connect(self.on_btn5_3_click)
        self.btn5_4.clicked.connect(self.on_btn5_4_click)
        self.btn5_5.clicked.connect(self.on_btn5_5_click)

    def on_btn1_1_click(self):
        Uncle = cv.imread("../Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg", cv.IMREAD_COLOR)
        print('Height:', Uncle.shape[0])
        print('width:', Uncle.shape[1])
        cv.imshow('Uncle_Roger',Uncle)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn1_2_click(self):
        # load img，convert to RGB
        Flower = cv.imread("../Dataset_opencvdl/Q1_Image/Flower.jpg", cv.IMREAD_COLOR)

        # Separate RGB color values
        (B, G, R) = cv.split(Flower)
        zeros = np.zeros(Flower.shape[:2], dtype='uint8') # Note that it is a single-channel image, do not write image.shape, it is a three-channel image
        actual_B = cv.merge([B, zeros, zeros])  # Green channel, pay attention to the order of B and the two zeros matrix, it must not be wrong~
        actual_G = cv.merge([zeros, G, zeros])
        actual_R = cv.merge([zeros, zeros, R])
        reActual_B = cv.resize(actual_B, (400, 300))
        reActual_G = cv.resize(actual_G, (400, 300))
        reActual_R = cv.resize(actual_R, (400, 300))


        cv.namedWindow('actualb.jpg')        # Create a named window
        cv.moveWindow('actualb.jpg', 0,30)  

        cv.namedWindow('actualg.jpg')        # Create a named window
        cv.moveWindow('actualg.jpg', 400,30)  

        cv.namedWindow('actualr.jpg')        # Create a named window
        cv.moveWindow('actualr.jpg', 800,30)  

        cv.imshow('actualb.jpg', reActual_B)
        cv.imshow('actualg.jpg', reActual_G)
        cv.imshow('actualr.jpg', reActual_R)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn1_3_click(self):
        Uncle = cv.imread("../Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg", cv.IMREAD_COLOR)

        Uncle__f = cv.flip(Uncle, 1)
        cv.namedWindow('Original Image')        # Create a named window
        cv.moveWindow('Original Image', 0,0) 
        cv.imshow('Original Image', Uncle)

        cv.namedWindow('Result')        # Create a named window
        cv.moveWindow('Result', 700,0) 
        cv.imshow('Result',Uncle__f)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn1_4_click(self):
        def Change(x):
            alpha = cv.getTrackbarPos('Blend', 'Blending')/100
            dst = cv.addWeighted(img1,alpha,img2,1-alpha,0)
            cv.imshow('Blending',dst)
        img1 = cv.imread( "../Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg", cv.IMREAD_COLOR )
        img2 = cv.flip(img1, 1)
        cv.namedWindow('Blending')
        cv.createTrackbar('Blend', 'Blending', 0,100,Change)
        # default image : flipped image
        dst = cv.addWeighted(img1,0,img2,1,0)
        cv.imshow('Blending',dst)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def on_btn2_1_click(self):
        Cat = cv.imread( "../Dataset_opencvdl/Q2_Image/Cat.png", cv.IMREAD_COLOR )
        median = cv.medianBlur(Cat,5)

        reCat = cv.resize(Cat, (600, 400))
        cv.namedWindow('Original Image')        # Create a named window
        cv.moveWindow('Original Image', 0,0)         
        cv.imshow('Original Image',reCat)

        remedian = cv.resize(median, (600, 400))
        cv.namedWindow('medianBlur')        # Create a named window
        cv.moveWindow('medianBlur', 600,0) 
        cv.imshow('medianBlur',remedian)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn2_2_click(self):
        Cat = cv.imread( "../Dataset_opencvdl/Q2_Image/Cat.png", cv.IMREAD_COLOR )
        image_gaussian_processed = cv.GaussianBlur(Cat,(3,3),1)

        reCat = cv.resize(Cat, (600, 400))
        cv.namedWindow('Original Image')        # Create a named window
        cv.moveWindow('Original Image', 0,0)         
        cv.imshow('Original Image',reCat)

        regaussian= cv.resize(image_gaussian_processed, (600, 400))
        cv.namedWindow('GaussianBlur')        # Create a named window
        cv.moveWindow('GaussianBlur', 600,0) 
        cv.imshow('GaussianBlur',regaussian)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn2_3_click(self):
        Cat = cv.imread( "../Dataset_opencvdl/Q2_Image/Cat.png", cv.IMREAD_COLOR )
        Bilateral = cv.bilateralFilter(Cat ,9,90,90)

        reCat = cv.resize(Cat, (600, 400))
        cv.namedWindow('Original Image')        # Create a named window
        cv.moveWindow('Original Image', 0,0)         
        cv.imshow('Original Image',reCat)

        reBilateral = cv.resize(Bilateral, (600, 400))
        cv.namedWindow('bilateralFilter')        # Create a named window
        cv.moveWindow('bilateralFilter', 600,0) 
        cv.imshow('bilateralFilter',reBilateral)
        cv.waitKey(0)
        cv.destroyAllWindows()                  
        

    def on_btn3_1_click(self):
        Chihiro = cv.imread( "../Dataset_opencvdl/Q3_Image/Chihiro.jpg", cv.IMREAD_COLOR )
        cv.namedWindow('Chihiro')        # Create a named window
        cv.moveWindow('Chihiro', 0,0) 
        cv.imshow('Chihiro',Chihiro)

        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
  
        image = mpimg.imread('../Dataset_opencvdl/Q3_Image/Chihiro.jpg')
        grayChihiro = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        '''
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()
        '''
        cv.namedWindow('Grayscale')        # Create a named window
        cv.moveWindow('Grayscale', 0,290) 
        cv.imshow('Grayscale',grayChihiro)

        gradGrayChihiro = signal.convolve2d(grayChihiro, gaussian_kernel, boundary='symm', mode='same') #卷積
        '''
        plt.imshow(grad, cmap=plt.get_cmap('gray'))
        plt.show()
        '''
        cv.namedWindow('Gaussian Blur')        # Create a named window
        cv.moveWindow('Gaussian Blur', 500,0) 
        cv.imshow('Gaussian Blur',grayChihiro)
        cv.waitKey(0)
        cv.destroyAllWindows()          

    def on_btn3_2_click(self):
        global cont3
        if cont3 < 3 :
            cont3 = cont3 | 1
            print("Generating Sobel X ")
            Chihiro = cv.imread( "../Dataset_opencvdl/Q3_Image/Chihiro.jpg", cv.IMREAD_COLOR )
    
            #3*3 Gassian filter
            x, y = np.mgrid[-1:2, -1:2]

            gaussian_kernel = np.exp(-(x**2+y**2))
            #Normalization
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
            image = mpimg.imread('../Dataset_opencvdl/Q3_Image/Chihiro.jpg')
            grayChihiro = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

            gradGrayChihiro = signal.convolve2d(grayChihiro, gaussian_kernel, boundary='symm', mode='same') #卷積

            sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
            sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)

            N = gradGrayChihiro.shape[0] #row 
            M = gradGrayChihiro.shape[1] #column

            image = np.pad(gradGrayChihiro, (1,1), 'edge')

            for i in range(1, N-1):
                for j in range(1, M-1):        
                    #Calculate gx and gy using Sobel (horizontal and vertical gradients)
                    gx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
                        (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
                        (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
                        (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
                        (sobelx[2][2] * image[i+1][j+1])

                    gy = (sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
                        (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
                        (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
                        (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
                        (sobely[2][2] * image[i+1][j+1])     

                    sobelxImage[i-1][j-1] = gx
                    sobelyImage[i-1][j-1] = gy

                    #Calculate the gradient magnitude
                    g = np.sqrt(gx * gx + gy * gy)
                    sobelGrad[i-1][j-1] = g
            cv.imwrite('custom_2d_convolution_gx.png',sobelxImage)
            Sobel_X= cv.imread( "custom_2d_convolution_gx.png", cv.IMREAD_COLOR )
            cv.imshow('Sobel X',Sobel_X)
            cv.waitKey(0)
            cv.destroyAllWindows() 
        else :
            Sobel_X= cv.imread( "custom_2d_convolution_gx.png", cv.IMREAD_COLOR )
            cv.imshow('Sobel X',Sobel_X)
            cv.waitKey(0)
            cv.destroyAllWindows()     

    def on_btn3_3_click(self):
        global cont3
        if cont3 < 3:
            cont3 = cont3 | 2
            print("Generating Sobel Y ")
            Chihiro = cv.imread( "../Dataset_opencvdl/Q3_Image/Chihiro.jpg", cv.IMREAD_COLOR )
    
            #3*3 Gassian filter
            x, y = np.mgrid[-1:2, -1:2]

            gaussian_kernel = np.exp(-(x**2+y**2))
            #Normalization
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
            image = mpimg.imread('../Dataset_opencvdl/Q3_Image/Chihiro.jpg')
            grayChihiro = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

            gradGrayChihiro = signal.convolve2d(grayChihiro, gaussian_kernel, boundary='symm', mode='same') #卷積

            sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
            sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)

            N = gradGrayChihiro.shape[0] #row 
            M = gradGrayChihiro.shape[1] #column

            image = np.pad(gradGrayChihiro, (1,1), 'edge')

            for i in range(1, N-1):
                for j in range(1, M-1):        
                    #Calculate gx and gy using Sobel (horizontal and vertical gradients)
                    gx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
                        (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
                        (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
                        (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
                        (sobelx[2][2] * image[i+1][j+1])

                    gy = (sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
                        (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
                        (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
                        (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
                        (sobely[2][2] * image[i+1][j+1])     

                    sobelxImage[i-1][j-1] = gx
                    sobelyImage[i-1][j-1] = gy

                    #Calculate the gradient magnitude
                    g = np.sqrt(gx * gx + gy * gy)
                    sobelGrad[i-1][j-1] = g        
            cv.imwrite('custom_2d_convolution_gy.png',sobelyImage)
            Sobel_Y= cv.imread( "custom_2d_convolution_gy.png", cv.IMREAD_COLOR )
            cv.imshow('Sobel Y',Sobel_Y)		       
            cv.waitKey(0)				       
            cv.destroyAllWindows()

        else :
            Sobel_Y= cv.imread( "custom_2d_convolution_gy.png", cv.IMREAD_COLOR )
            cv.imshow('Sobel Y',Sobel_Y)
            cv.waitKey(0)
            cv.destroyAllWindows()             

    def on_btn3_4_click(self):
        global cont3
        print(cont3)
        if cont3 == 0 :
            print("You must be generated Sobel X ans Sobel Y ")
        elif cont3 == 1:
            print("You must be generated Sobel Y ")
        elif cont3 == 2:
            print("You must be generated Sobel X ")            
        else :
            cv.imwrite('custom_2d_convolution_Grad.png',sobelGrad)
            Magnitude= cv.imread( "custom_2d_convolution_Grad.png", cv.IMREAD_COLOR )
            cv.imshow('Magnitude',Magnitude)				       
            cv.waitKey(0)
            cv.destroyAllWindows()


    def on_btn4_1_click(self):
        # read the image
        img = cv.imread('../Dataset_opencvdl/Q4_Image/Parrot.png')

        # read the transform data from ui
        edtAngle = float(self.edtAngle.text())
        edtScale = float(self.edtScale.text())
        edtTx = float(self.edtTx.text())
        edtTy = float(self.edtTy.text())

        # making translate matrix
        H = np.float32([[1,0,edtTx],[0,1,edtTy]])

        # translate the small squared image
        rows,cols = img.shape[:2]
        Translate_img = cv.warpAffine(img,H,(rows,cols))

        # making rotate and scale matrix
        rows,cols = Translate_img.shape[:2]
        M = cv.getRotationMatrix2D((160+edtTx,84+edtTy),edtAngle,edtScale)

        #rotate and scale the small squared image
        result = cv.warpAffine(Translate_img,M,(rows,cols))

        #show the result
        cv.imshow('Original Image', img)
        cv.imshow('Rotation + Translate + Scale Imag',result)
        cv.waitKey(0)
        cv.destroyAllWindows()
    def on_btn5_1_click(self):
        # load dataset
        (trainX, trainy), (testX, testy) = cifar10.load_data()
        # summarize loaded dataset
        print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
        print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
        fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(17, 8))
        index = 0
        for i in range(2):
            for j in range(5):
                axes[i,j].set_title(labels[trainy[index][0]])
                axes[i,j].imshow(trainX[index])
                axes[i,j].get_xaxis().set_visible(False)
                axes[i,j].get_yaxis().set_visible(False)
                index += 1
        plt.show()


    def on_btn5_2_click(self):
        print('hyperparameters: ')
        print('batch size: 64')
        print('learning rate: ',K.eval(model.optimizer.lr))
        print('optimizer: SGD')
        
    def on_btn5_3_click(self):
        model.summary()

    def on_btn5_4_click(self):
        acc = cv.imread("0-main.py_plot.png", cv.IMREAD_COLOR)
        cv.imshow('acc',acc)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_btn5_5_click(self):
        x_img_train_normalize = x_img_train.astype('float32') / 255.0
        x_img_test_normalize = x_img_test.astype('float32') / 255.0

        prediction = model.predict_classes(x_img_test_normalize)
        Predicted_Probability = model.predict(x_img_test_normalize)
        label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
        i = int(self.SpinBox5.text())    

        print('label:',label_dict[y_label_test[i][0]],'predict:',label_dict[prediction[i]])
        preArr = []
        for j in range(10):
            preArr.append('%1.9f'% (Predicted_Probability[i][j]))
        preArr = list(map(float, preArr))    
        y=np.arange(0, 1, 0.2)
        plt.figure(figsize=(10,5)) 
        plt.yticks(y)   
        plt.bar(labels, preArr, label = 'acc')

        plt.figure(figsize=(2,2))
        plt.imshow(np.reshape(x_img_test[i],(32, 32,3)))
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
