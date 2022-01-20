# -*- coding: utf-8 -*-
from doctest import OutputChecker
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QApplication, QLabel
from tkinter import filedialog
# model inference
import model.scalp_model as scalp_model
import torch
import os
import sys
# output gui
from GUI.view_output import main as view_output
import numpy

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")  # background color
        
        # program title & introduction
        self.logo = QPixmap('./GUI/logo_intro.png')
    
        intro = "*Oh My Head*는 두피분석 전문 프로그램입니다.\n지금 바로 당신의 두피 상태를 확인해보세요!\n\n아래는 두피 촬영 예시 이미지입니다."
        self.introduction = QLabel(intro, self)
        font = self.introduction.font()
        font.setPointSize(13)
        self.introduction.setFont(font)
        
        # get example image
        self.state0_img = QPixmap('./GUI/scalp_example/양호.jpg').scaledToHeight(100)
        self.state1_img = QPixmap('./GUI/scalp_example/경증.jpg').scaledToHeight(100)
        self.state2_img = QPixmap('./GUI/scalp_example/중등도.jpg').scaledToHeight(100)
        self.state3_img = QPixmap('./GUI/scalp_example/중증.jpg').scaledToHeight(100)
        
        self.big_box = QVBoxLayout()
        self.filepath = ''
        self.initUI()
        
    def initUI(self):
        # get image
        self.img0 = QLabel(self)
        self.img1 = QLabel(self)
        self.img2 = QLabel(self)
        self.img3 = QLabel(self)
        
        self.img0.setPixmap(self.state0_img)
        self.img1.setPixmap(self.state1_img)
        self.img2.setPixmap(self.state2_img)
        self.img3.setPixmap(self.state3_img)
        self.img0.setAlignment(Qt.AlignCenter)
        self.img1.setAlignment(Qt.AlignCenter)
        self.img2.setAlignment(Qt.AlignCenter)
        self.img3.setAlignment(Qt.AlignCenter)
        
        # show image
        eximg_box = QGridLayout()
        eximg_box.addWidget(self.img0,0,0)
        eximg_box.addWidget(self.img1,0,1)
        eximg_box.addWidget(self.img2,1,0)
        eximg_box.addWidget(self.img3,1,1)
        
        # get logo
        self.label = QLabel(self)
        self.label.setPixmap(self.logo)
        self.label.setAlignment(Qt.AlignCenter)
        
        # show introduction 
        intro_box = QGridLayout() # intro_box = QVBoxLayout()
        intro_box.addWidget(self.label,0,0) # logo
        intro_box.addWidget(self.introduction,1,0) # 소개글
        
        # button event
        button1 = QPushButton("두피 이미지 불러오기", self)
        button1.clicked.connect(self.button1_clicked)
        
        # show all in big_box
        self.big_box.addLayout(intro_box)
        self.big_box.addLayout(eximg_box)
        self.big_box.addWidget(button1)
        
        self.setLayout(self.big_box)
        self.setWindowTitle('Welcome!')
        self.setGeometry(550, 200, 200, 300)
        self.show()
        
    def button1_clicked(self): # 두피 이미지 불러오기 버튼
        # get user image
        self.filepath = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(('jpg files', '*.jpg'),
    ('png files', '*.png'),('all files', '*.*')))
        
        # show filename
        self.user_img = QLabel(self.filepath, self)
        font = self.user_img.font()
        font.setPointSize(8)
        self.user_img.setFont(font)
        self.big_box.addWidget(self.user_img)
        
        button2 = QPushButton("분석하기", self)
        button2.clicked.connect(self.button2_clicked)
        self.big_box.addWidget(button2)
        
    def button2_clicked(self): # 분석하기 버튼
        # model inference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        PATH = os.getcwd() + "/Scalp_model_parameters/"
        model = scalp_model.load_model_trained(device, PATH)
        self.output = scalp_model.test_model(self.filepath, model, device).view(-1) # tensor
        
        # inference 값 중 음수 값 처리 
        # 0 = 양호, 0.33 = 경증, 0.66 = 중증도, 1 = 중증
        # 음수는 0(양호)에 가까우므로, 0으로 치환한다.
        for i in range(0,6) :
            if self.output[i] < 0 : self.output[i] = 0
            
        # save image path & tensor
        imgpath = self.filepath
        f = open('./output_info.txt','w')
        f.write(imgpath)
        f.close
        inference_result = self.output.detach().numpy()
        numpy.save('./inference_result',inference_result)
        
        # button event
        button3 = QPushButton("결과 보러 가기", self)
        button3.clicked.connect(QCoreApplication.instance().quit) # close
        self.big_box.addWidget(button3)    


if __name__ == '__main__':
   app = QtWidgets.QApplication(sys.argv)
   ex = MyWidget()
   ex.show()
   app.exec_()
   
   f = open('./output_info.txt','r') # output information(img path, tensor)
   filepath = f.read()
   inference_result = numpy.load('./inference_result.npy')
   inference_result = torch.from_numpy(inference_result)
   
   view_output(filepath, inference_result)
   sys.exit()