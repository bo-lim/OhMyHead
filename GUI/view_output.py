import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic, QtGui 
import torch 

### PATH ###
# current_path = ./OHMYHAIR
current_path = os.getcwd()

### Connect to ui file ###
form_class = uic.loadUiType(os.path.join(current_path, "GUI/output.ui"))[0]

### Disease description ###
description = []    
with open(os.path.join(current_path, "GUI/disease_explained.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        description.append(line)

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

### GUI Output ###
class WindowClass(QMainWindow, form_class) :
    def __init__(self, image_path, pred_tensor) :
        super().__init__()
        self.setupUi(self)
        ### Prediction of model ###
        prediction_vec = pred_tensor
        prediction_list = ['미세각질', '피지과다', '모낭사이홍반', '모낭홍반농포', '비듬', '탈모']
        prediction = prediction_list[torch.argmax(prediction_vec)]
        max_pro = torch.max(prediction_vec).item()
        # 0=양호, 0.33=경증, 0.66=중등도, 1.00=중증
        if max_pro < 0.33/2:
            severity = '양호'
        elif max_pro >= 0.33/2 or max_pro < (0.33+0.66)/2:
            severity = '경증'
        elif max_pro >= (0.33+0.66)/2 or max_pro < (1+0.66)/2:
            severity = '중등도'
        else:
            severity = '중증'

        # Set the window title
        self.setWindowTitle("Oh My Head - CUAI winter project - Healthcare team")

        # Set size of window
        # self.setFixedSize(1280, 600)
        self.setFixedSize(1400, 710)

        # Title image
        self.qPixmapFileVar_title = QPixmap()
        self.qPixmapFileVar_title.load(os.path.join(current_path, 'GUI/logo_output.png'))
        self.qPixmapFileVar_title = self.qPixmapFileVar_title.scaledToWidth(200)
        self.title_box.setPixmap(self.qPixmapFileVar_title)

        # Scalp text
        self.scalp_txt.setText("당신의 두피는 *{}*인 것으로 보입니다.".format(prediction))
        self.scalp_txt.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        # Scalp image
        self.qPixmapFileVar_scalp = QPixmap()
        self.qPixmapFileVar_scalp.load(os.path.join(current_path, image_path))
        self.qPixmapFileVar_scalp = self.qPixmapFileVar_scalp.scaledToWidth(300)
        self.my_scalp.setPixmap(self.qPixmapFileVar_scalp)

        # Model prediction
        self.result_txt.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.model_prediction.setText("당신의 두피 두피 증상 {}에 대한 심각도는 {}입니다. 수치상으로는 {}이며 기준은 다음과 같습니다.".format(
                                        prediction, severity, max_pro))
        self.model_prediction2.setText("(0=양호, 0.33=경증, 0.66=중등도, 1.00=중증)")
        self.model_prediction.setFont(QtGui.QFont("Arial", 15))
        self.model_prediction2.setFont(QtGui.QFont("Arial", 15))


        # Disease description
        self.result_txt2.setText(prediction + " 관련 정보")
        self.result_txt2.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.disease.setText(description[prediction_list.index(prediction)].split(':')[1].strip())
        self.disease.setFont(QtGui.QFont("Arial", 15))

        # Survey title
        self.analysis_title.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        ## Survey figures
        # shampoo
        self.qPixmapFileVar_title2 = QPixmap()
        self.qPixmapFileVar_title2.load(os.path.join(current_path, f"GUI/figure/{prediction}_샴푸사용빈도.png"))
        self.qPixmapFileVar_title2 = self.qPixmapFileVar_title2.scaledToWidth(720)
        self.shampoo_figure.setPixmap(self.qPixmapFileVar_title2)
        # perm
        self.qPixmapFileVar_title3 = QPixmap()
        self.qPixmapFileVar_title3.load(os.path.join(current_path, f"GUI/figure/{prediction}_펌주기.png"))
        self.qPixmapFileVar_title3 = self.qPixmapFileVar_title3.scaledToWidth(720)
        self.perm_figure.setPixmap(self.qPixmapFileVar_title3)
        # dye
        self.qPixmapFileVar_title4 = QPixmap()
        self.qPixmapFileVar_title4.load(os.path.join(current_path, f"GUI/figure/{prediction}_염색주기.png"))
        self.qPixmapFileVar_title4 = self.qPixmapFileVar_title4.scaledToWidth(720)
        self.dye_figure.setPixmap(self.qPixmapFileVar_title4)

        
def main(image_path, pred_tensor):
    app = QApplication(sys.argv) 
    myWindow = WindowClass(image_path, pred_tensor)
    myWindow.show()
    app.exec_()