import cv2
import tensorflow as tf
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cam = cv2.VideoCapture(0)

BASE_DIR = os.path.abspath(os.getcwd())
CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint")

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(16, 3, activation='relu',input_shape=(478,3)),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='adam',metrics=['accuracy'])
model.load_weights(CHECKPOINT_PATH+'\landmark_base.h5')

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
      max_num_faces=1,
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
    return landmark

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cam.isOpened():
    success, image = cam.read()
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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())
        # Flip the image horizontally for a selfie-view display.
        image_landmark = clearLandmark(face_landmarks)
        image_for_model = np.expand_dims(image_landmark, axis=0)
        prediction = model.predict(image_for_model,verbose = 0)
        image = cv2.flip(image, 1)
        if (prediction[0]< 0.5):
            image = cv2.putText(image, 'Droopy {:01f} %'.format((1-prediction[0][0]*2)*100), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            image = cv2.putText(image, 'Normal {:01f} %'.format((prediction[0][0])*100), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('MediaPipe Face Mesh', image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()