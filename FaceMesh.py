import cv2
import mediapipe as mp
import time

v = cv2.VideoCapture(0)
# v = cv2.VideoCapture("CharliePuth.webm")
# v = cv2.VideoCapture("BlankSpace.mp4")

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

Drawing_Specifications = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

while True:
    ret, frame = v.read()

    # converted_image = cv2.imread(frame, 0)
    # converted_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # results = faceMesh.process(converted_image)
    Faces = faceMesh.process(frame)


    if Faces.multi_face_landmarks:
        for faceLms in Faces.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL,Drawing_Specifications,Drawing_Specifications)
            landmarks = faceLms

    # if landmarks == faceLms :
    #     print('Same')



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

    # frame = cv2.resize(frame, (800, 440))

    cv2.imshow('Live Cam', frame)

    # cv2.waitKey(1)
    if cv2.waitKey(1) == 13:  # Until the Enter button is touched. 13->ENTER
        break

print(landmarks)

# cv2.waitKey(0)
v.release()
cv2.destroyAllWindows()
