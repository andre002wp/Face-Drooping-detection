from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QThread,pyqtSignal
import sys
import os
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
max_face = 3
BASE_DIR = os.path.abspath(os.getcwd())
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint")
BASELINE = 0.3

class Ui(QtWidgets.QMainWindow):

    img = None
    camState = False
    model = tf.keras.models.load_model(os.path.join(CHECKPOINT_PATH, 'landmark_base.h5'))

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Face_droop.ui', self)
        
        self.tabWidget.currentChanged.connect(self.tabChanged)
        self.btnOpenFile = self.findChild(QtWidgets.QPushButton, 'btn_select_img')
        self.btnOpenFile.clicked.connect(self.loadImage)

        self.btnOpenCam = self.findChild(QtWidgets.QPushButton, 'btn_opencam')
        self.btnOpenCam.clicked.connect(self.setCam)

        self.txtImage = self.findChild(QtWidgets.QTextEdit, 'image_desc')
        self.txtImage.setReadOnly(True)

        self.image_ui = self.findChild(QtWidgets.QLabel, 'img_detect')

        self.camera_ui = self.findChild(QtWidgets.QLabel, 'camera_detect')

        self.cam_port = self.findChild(QtWidgets.QComboBox, 'port_selector')
        self.cam_port.addItems(self.DetectCamPort())
        self.cam_port.setCurrentIndex(0)
        self.cam_port.activated.connect(self.clearUI)

        self.camWorker = CamWorker()
        self.camWorker.setModel(self.model)

        self.show()

    def clearUI(self):
        self.camera_ui.clear()

    def DetectCamPort(self):
        self.valid_cams = []
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                print('Found video source at : ', i)
                self.valid_cams.append(str(i))
            cap.release()
        return self.valid_cams

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.showPrediction()
        return super().resizeEvent(a0)

    def tabChanged(self):
        print("Tab Changed to : ",self.tabWidget.currentIndex())

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

    def setCam(self):
        if(self.camState):
            print("Camera Closed")
            self.camState = False
            self.btnOpenCam.setText("Nyalakan Kamera")
            self.camWorker.stop()
        else:
            print("Camera Opened")
            try:
                self.camState = True
                self.btnOpenCam.setText("Matikan Kamera")
                self.camWorker.setPort(int(self.cam_port.currentText()))
                self.camWorker.start()
                self.camWorker.ImageSignal.connect(self.updateCamUI)
            except:
                self.camState = False
                self.btnOpenCam.setText("Nyalakan Kamera")
                print("Error Opening Camera")
                self.camWorker.stop()
                self.updateCamUI(np.zeros((480,640,3),dtype=np.uint8))

    def updateCamUI(self,img):
        self.camera_ui.setPixmap(self.rezizeandShow(img,self.camera_ui))
        

    def loadImage(self):
        #hack
        if self.img is not None:
            try:
                _tmp_img = self.img
                _tmp_bbox = self.img_bbox
            except:
                pass
        # This is executed when the button is pressed
        print('btn Open File Pressed')
        openFileDialog = QFileDialog.getOpenFileName(self,"select Image File",os.getcwd(),"Image Files (*.jpg *.gif *.bmp *.png *.tiff *.jfif)")
        fileimg = openFileDialog[0]

        try:
            if(len(fileimg)>4):
                self.img = cv2.imread(fileimg,cv2.IMREAD_UNCHANGED)
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                img_inference = self.img.copy()
                rect_min,rect_max,image_landmark = getFaceLandmark(fileimg)
                image_for_model = np.expand_dims(image_landmark, axis=0)
                
                cv2.rectangle(img_inference, rect_min, rect_max, (255, 255, 0), self.setBoxSize(rect_max[0]-rect_min[0]))
                prediction = self.model.predict(image_for_model,verbose = 0)

                self.img_bbox = (rect_min,rect_max)
                self.img_prediction = prediction
                if (prediction[0][0]< BASELINE):
                    print("Droopy Confidence {:02f} %".format((1-(prediction[0][0]/(BASELINE-0)))*100))
                else:
                    print("Normal Confidence {:02f} %".format((((prediction[0][0]-BASELINE)/(1-BASELINE)))*100))

                self.showPrediction()
        except:
            try:
                self.img = _tmp_img
                self.img_bbox = _tmp_bbox
            except:
                pass
            QMessageBox.about(self, "Error", "Gagal Memuat Gambar")

    def setBoxSize(self,size):
        size = int(size//100)
        if size<1:
            size = 1
        elif size>10:
            size = 10
            
        return size
    
    def showPrediction(self):
        if(self.img is not None):
            img_inference = self.img.copy()
            rect_min,rect_max = self.img_bbox
            cv2.rectangle(img_inference, rect_min, rect_max, (255, 255, 0), 3)
            prediction = self.img_prediction
            if (prediction[0][0]< BASELINE):
                _txt = "Keterangan :\nDroopy\nConfidence :\n{:02f} %".format((1-(prediction[0][0]/(BASELINE-0)))*100)
                self.txtImage.setText(_txt)
            else:
                _txt = "Keterangan :\nTidak Droopy\nConfidence :\n{:02f} %".format((((prediction[0][0]-BASELINE)/(1-BASELINE)))*100)
                self.txtImage.setText(_txt)
            self.image_ui.setPixmap(self.rezizeandShow(img_inference,self.image_ui))

class CamWorker(QThread):
    ImageSignal = pyqtSignal(np.ndarray)
    port = 0
    model = None
    cam = None

    def run(self):
        self.ThreadActive = True
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_face,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while self.ThreadActive:
                success, image = self.cam.read()
                image = cv2.flip(image, 1)
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
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
                        cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)
                        image_landmark = clearLandmark(face_landmarks)
                        image_for_model = np.expand_dims(image_landmark, axis=0)
                        prediction = self.model.predict(image_for_model,verbose = 0)
                        if (prediction[0][0]< BASELINE):
                            image = cv2.putText(image, 'Droopy {:01f} %'.format((1-(prediction[0][0]/(BASELINE-0)))*100), (cx_max-cx_min, cy_max-cy_min), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            image = cv2.putText(image, 'Tidak Droopy {:01f} %'.format((((prediction[0][0]-BASELINE)/(1-BASELINE)))*100), (cx_max-cx_min, cy_max-cy_min), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
                self.ImageSignal.emit(image)

    def setModel(self,model):
        self.model = model

    def setPort(self,port):
        self.port = port
        if self.cam is not None and self.cam.isOpened():
            self.cam.release()
        self.cam = cv2.VideoCapture(self.port)
    
    def stop(self):
        self.ThreadActive = False
        self.cam.release()
        self.quit()

def clearLandmark(data):
        cleaned_data = []
        for i in range(len(data.landmark)):
            valx = data.landmark[i].x
            valy = data.landmark[i].y
            valz = data.landmark[i].z
            cleaned_data.append([valx,valy,valz])
        return cleaned_data

def getFaceLandmark(fileitem):
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
            landmark = clearLandmark(face_landmarks)
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

app = QtWidgets.QApplication(sys.argv)
window = Ui()
inference_app = app.exec_()
sys.exit(inference_app)