import time

import cv2
import mediapipe

# Capturing the Video form the Live Cam.
# V = cv2.VideoCapture(0)

# Capturing the Video provided with the path of the file.
V = cv2.VideoCapture("BlankSpace.mp4")

PreviousTime = 0

FaceDetection = mediapipe.solutions.face_detection
BoundingBoxes = mediapipe.solutions.drawing_utils
fc_d = FaceDetection.FaceDetection()

while True:
    success, frame = V.read()

    frame = cv2.resize(frame, (800, 440))

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = fc_d.process(RGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            BoundingBoxes.draw_detection(frame, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Formulating the Frames per second
    CurrentTime = time.time()
    FPS = 1 / (CurrentTime - PreviousTime)
    PreviousTime = CurrentTime
    cv2.putText(frame, f'FPS:{int(FPS)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    cv2.imshow("VIDEO or LiveCam", frame)
    # Reduce the Frame Rate by increasing the waitkey.
    if cv2.waitKey(1) == 13:
        break
