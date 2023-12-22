###############################################################################################################################
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import * 
from PyQt6.uic import loadUi
from PyQt6.QtGui import QPixmap , QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt

import os
import sys
import cv2
#import requests
from gender.test import predict 
from gender.video import detect

fileName = ""
#cửa sổ main
class Signature_w(QMainWindow):
    def __init__(self):
        super(Signature_w,self).__init__()
        uic.loadUi('gender.ui',self)
        self.open.clicked.connect(self.openfile)
        self.predict.clicked.connect(self.recognize)
        self.detect.clicked.connect(self.detect_video)

    def openfile(self) :
        global fileName 
        path = QFileDialog.getOpenFileName(self,"Open File", "","All Files (*);;Png Files (*.png)")
        if path:
            fileName = path[0]
            pixmap = QPixmap(fileName)
            self.input.setPixmap(pixmap)
            self.input.setScaledContents(True)   

    def recognize(self):
        global fileName 
        if fileName != "" :
            res = predict(fileName)   

            self.label.clear()
            self.label.setText(res) 

            self.image = cv2.imread(fileName)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detect_and_display_faces() 

    def detect_and_display_faces(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        global fileName 

        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pixmap = QPixmap(fileName)
            painter = QPainter(pixmap)
            
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255))
            pen.setWidth(7)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)
            
            painter.end() 
            
            self.input.setPixmap(pixmap)
            self.input.setScaledContents(True)
        

    def detect_video(self) :
        path = QFileDialog.getOpenFileName(self,"Open File", "","All Files (*);;Mp4 Files (*.mp4)")
        if path:
            detect(path[0]) 

#xử lí
app = QApplication(sys.argv) 
widget = QtWidgets.QStackedWidget() 
Signature_f = Signature_w()
widget.addWidget(Signature_f)
widget.setCurrentIndex(0)
widget.setWindowTitle("Gender dectection")
widget.setFixedHeight(617)
widget.setFixedWidth(434)
widget.show()
app.exec() 
