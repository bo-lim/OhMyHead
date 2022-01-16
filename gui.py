# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel
from tkinter import filedialog

# model inference
import model.scalp_model as scalp_model
import model.scalp_dataset as scalp_dataset
import torch
import os
import sys

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")  # background color
        
        # program title & introduction
        self.logo = QPixmap('./GUI/logo.png').scaledToWidth(180)
    
        intro = "Oh My Hair는 전문 두피 분석 프로그램입니다.\n지금 바로 당신의 두피 상태를 확인해보세요!\n아래는 두피 종류 예시 이미지입니다."
        self.introduction = QLabel(intro, self)
        font = self.introduction.font()
        font.setPointSize(10)
        self.introduction.setFont(font)
        
        # get example image
        self.state0_img = QPixmap('./GUI/scalp_example/양호.jpg').scaledToHeight(70)
        self.state1_img = QPixmap('./GUI/scalp_example/경증.jpg').scaledToHeight(70)
        self.state2_img = QPixmap('./GUI/scalp_example/중등도.jpg').scaledToHeight(70)
        self.state3_img = QPixmap('./GUI/scalp_example/중증.jpg').scaledToHeight(70)
        
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
        self.setGeometry(200, 300, 200, 350)
        self.show()
        
    def button1_clicked(self):
        # get user image
        self.filepath = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(('jpg files', '*.jpg'),
    ('png files', '*.png'),('all files', '*.*')))
        
        # show filename
        self.user_img = QLabel(self.filepath, self)
        font = self.user_img.font()
        font.setPointSize(8)
        self.user_img.setFont(font)
        self.big_box.addWidget(self.user_img)
        
        button2 = QPushButton("분석결과 확인하기", self)
        button2.clicked.connect(self.button2_clicked)
        self.big_box.addWidget(button2)
        
        
    def button2_clicked(self):
        # print(self.filepath)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        PATH = os.getcwd() + "/Scalp_model_parameters/"
        model = scalp_model.load_model_trained(device, PATH)
        scalp_model.test_model(self.filepath, model, device)

if __name__ == '__main__':
   app = QtWidgets.QApplication(sys.argv)
   ex = MyWidget()
   ex.show()
   sys.exit(app.exec_())