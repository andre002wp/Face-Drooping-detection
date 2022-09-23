import cv2
import tensorflow as tf
import os
import mediapipe as mp
import numpy as np
from mtcnn.mtcnn import MTCNN
from FileImporter import openFiles

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
max_face = 3

BASE_DIR = os.path.abspath(os.getcwd())
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint")

model = tf.keras.models.load_model(os.path.join(CHECKPOINT_PATH, 'landmark_base.h5'))

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
            rec_max = (cx_max, cy_max)
    return rect_min,rec_max,landmark

###########################
filepath = openFiles()
# filepath = os.path.join(BASE_DIR,'stroke-normal_dataset_processed\Train\\normal\download.png')
###########################

# # extract a single face from a given photograph
# def extract_face(filename, required_size=(224, 224)):
# 	# load image from file
# 	pixels = cv2.imread(filename, cv2.COLOR_BGR2RGB)
# 	pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
# 	# create the detector, using default weights
# 	detector = MTCNN()
# 	# detect faces in the image
# 	results = detector.detect_faces(pixels)
# 	# extract the bounding box from the first face
# 	x1, y1, width, height = results[0]['box']
# 	x2, y2 = x1 + width, y1 + height
# 	# extract the face
# 	face = pixels[y1:y2, x1:x2]
# 	# resize pixels to the model size
# 	image = cv2.resize(face, required_size)
# 	return image
# # load the photo and extract the face
# face_extract = extract_face(filepath)

im = cv2.imread(filepath)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
rect_min,rec_max,image_landmark = getFaceLandmark(filepath)
image_for_model = np.expand_dims(image_landmark, axis=0)
cv2.rectangle(im, rect_min, rec_max, (255, 255, 0), 2)
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
if(im.shape[0]>960 or im.shape[1]>640):
    im = cv2.resize(im, (int(im.shape[1]/2), int(im.shape[0]/2)))
prediction = model.predict(image_for_model,verbose = 0)
baseline = 0.3
print("nilai prediksi = ",prediction[0][0])
if (prediction[0][0]< baseline):
    print("Droopy Confidence {:02f} %".format((1-(prediction[0][0]/(baseline-0)))*100))
else:
    print("Tidak Droopy Confidence {:02f} %".format((((prediction[0][0]-baseline)/(1-baseline)))*100))

cv2.imshow("results", im)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

