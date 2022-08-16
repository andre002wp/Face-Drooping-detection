import cv2
import tensorflow as tf
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
max_face = 3
cam = cv2.VideoCapture(0)

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

with mp_face_mesh.FaceMesh(
    max_num_faces=max_face,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cam.isOpened():
    success, image = cam.read()
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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_tesselation_style())
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
        baseline = 0.3
        if (prediction[0][0]< baseline):
            # image = cv2.putText(image, 'Droopy {:01f} %'.format(prediction[0][0]), (cx_min, cy_max), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, 'Droopy {:01f} %'.format((1-(prediction[0][0]/(baseline-0)))*100), (cx_max-cx_min, cy_max-cy_min), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # image = cv2.putText(image, 'Normal {:01f} %'.format(prediction[0][0]), (cx_min, cy_max), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, 'Normal {:01f} %'.format((((prediction[0][0]-baseline)/(1-baseline)))*100), (cx_max-cx_min, cy_max-cy_min), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.namedWindow('Face Droop Detection',cv2.WINDOW_NORMAL)
    cv2.imshow('Face Droop Detection', image)
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty('Face Droop Detection',cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()