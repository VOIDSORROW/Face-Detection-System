import time

import cv2
import mediapipe


class FaceDetector():
    def __int__(self, scr=0.5):
        self.scr = scr

        self.FaceDetection = mediapipe.solutions.face_detection
        self.BoundingBoxes = mediapipe.solutions.drawing_utils
        self.fc_d = self.FaceDetection.FaceDetection()

    def FindFaces(self, frame=True):
        frame = cv2.resize(frame, (800, 440))

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.fc_d.process(RGB)

        faces = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                self.BoundingBoxes.draw_detection(frame, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2)
                faces.append([id, detection.score])
        return frame, faces


def main():
    # V = cv2.VideoCapture(0)
    V = cv2.VideoCapture("BlankSpace.mp4")

    PreviousTime = 0

    detector = FaceDetector()

    while True:
        success, frame = V.read()
        frame, faces = detector.FindFaces(frame)
        CurrentTime = time.time()
        FPS = 1 / (CurrentTime - PreviousTime)
        PreviousTime = CurrentTime
        cv2.putText(frame, f'FPS:{int(FPS)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        cv2.imshow("VIDEO or LiveCam", frame)
        if cv2.waitKey(1) == 13:
            break


if __name__ == "__main__":
    main()
