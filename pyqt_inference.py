from tkinter import Y
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QImage,QPixmap
import sys
import os
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
from mtcnn.mtcnn import MTCNN

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
max_face = 3

BASE_DIR = os.path.abspath(os.getcwd())
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint")

class Ui(QtWidgets.QMainWindow):

    img = None
    model = tf.keras.models.load_model(os.path.join(CHECKPOINT_PATH, 'landmark_base.h5'))

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Face_droop.ui', self)
        
        self.tabWidget.currentChanged.connect(self.tabChanged)
        self.btnOpenFile = self.findChild(QtWidgets.QPushButton, 'btn_select_img')
        self.btnOpenFile.clicked.connect(self.loadImage)

        self.btnOpenCam = self.findChild(QtWidgets.QPushButton, 'btn_opencam')
        self.btnOpenCam.clicked.connect(self.openCam)

        self.txtImage = self.findChild(QtWidgets.QTextEdit, 'image_desc')
        self.txtImage.setReadOnly(True)

        self.image_ui = self.findChild(QtWidgets.QLabel, 'img_detect')
        self.image_ui

        self.show()

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.showPrediction()
        return super().resizeEvent(a0)

    def tabChanged(self):
        print("Tab Changed to : ",self.tabWidget.currentIndex())
    
    def clearLandmark(self,data):
        cleaned_data = []
        for i in range(len(data.landmark)):
            valx = data.landmark[i].x
            valy = data.landmark[i].y
            valz = data.landmark[i].z
            cleaned_data.append([valx,valy,valz])
        return cleaned_data

    def getFaceLandmark(self,fileitem):
        with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_face,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
            image = cv2.imread(fileitem)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                return None
            for face_landmarks in results.multi_face_landmarks:
                landmark = self.clearLandmark(face_landmarks)
                h, w, c = image.shape
                cx_min =  w
                cy_min = h
                cx_max = cy_max= 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                rect_min = (cx_min, cy_min)
                rect_max = (cx_max, cy_max)
        return rect_min,rect_max,landmark

    def rezizeandShow(self,img,ui):  
        img_temp = img.copy()
        try:
            y,x,color = img_temp.shape
        except:
            y,x = img.shape
            color = 1
        
        if(y>x):
            cv2.rotate(img_temp, cv2.ROTATE_90_CLOCKWISE)

        bytesPerLine = 3 * x
        if(color== 1):
            _pmap = QImage(img_temp, x, y, x, QImage.Format_Grayscale8)
        else:
            _pmap = QImage(img_temp, x, y, bytesPerLine, QImage.Format_RGB888)
        _pmap = QPixmap(_pmap)

        (width,height) = ui.size().width(),ui.size().height()
        imgresized = _pmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        # test = _pmap.toImage()
        # test.pixel(0,0)
        # for i in range(y):
        #     for j in range(x):
        #         print(f"{i,j} : {QtGui.qRed(test.pixel(i,j))},{QtGui.qGreen(test.pixel(i,j))},{QtGui.qBlue(test.pixel(i,j))}")
        
        
        return imgresized

    def openCam(self):
        pass

    def loadImage(self):
        #hack
        _tmp_img = self.img
        _tmp_bbox = self.img_bbox
        # This is executed when the button is pressed
        print('btn Open File Pressed')
        openFileDialog = QFileDialog.getOpenFileName(self,"select Image File",os.getcwd(),"Image Files (*.jpg *.gif *.bmp *.png *.tiff *.jfif)")
        fileimg = openFileDialog[0]

        try:
            if(len(fileimg)>4):
                self.img = cv2.imread(fileimg,cv2.IMREAD_UNCHANGED)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                img_inference = self.img.copy()
                rect_min,rect_max,image_landmark = self.getFaceLandmark(fileimg)
                image_for_model = np.expand_dims(image_landmark, axis=0)
                cv2.rectangle(img_inference, rect_min, rect_max, (255, 255, 0), 3)
                prediction = self.model.predict(image_for_model,verbose = 0)

                self.img_bbox = (rect_min,rect_max)
                self.img_prediction = prediction
                self.showPrediction()
        except:
            self.img = _tmp_img
            self.img_bbox = _tmp_bbox
            QMessageBox.about(self, "Error", "Gagal Memuat Gambar")
    
    def showPrediction(self):
        if(self.img is not None):
            img_inference = self.img.copy()
            rect_min,rect_max = self.img_bbox
            cv2.rectangle(img_inference, rect_min, rect_max, (255, 255, 0), 3)
            prediction = self.img_prediction
            baseline = 0.3
            if (prediction[0][0]< baseline):
                print("Droopy Confidence {:02f} %".format((1-(prediction[0][0]/(baseline-0)))*100))
                _txt = "Keterangan :\nDroopy\nConfidence :\n{:02f} %".format((1-(prediction[0][0]/(baseline-0)))*100)
                self.txtImage.setText(_txt)
            else:
                _txt = "Keterangan :\nTidak Droopy\nConfidence :\n{:02f} %".format((((prediction[0][0]-baseline)/(1-baseline)))*100)
                self.txtImage.setText(_txt)
            self.image_ui.setPixmap(self.rezizeandShow(img_inference,self.image_ui))


app = QtWidgets.QApplication(sys.argv)
window = Ui()
inference_app = app.exec_()
sys.exit(inference_app)