import cv2
import mediapipe as mp
from math import hypot


class HandDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = mp.solutions.hands.Hands(max_num_hands=1)
        self.listLm = list()
    
    def findLandMarks(self, img, result):
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    self.listLm.append((id, cx, cy))
        
        return self.listLm
    
    def findDist(self, img, p1, p2):
        if not self.listLm:
            return -1

        x1, y1 = self.listLm[p1][1:]
        x2, y2 = self.listLm[p2][1:]

        dist = round(hypot(max(x1, x2) - min(x1, x2), max(y1, y2) - min(y1, y2)))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(img, str(dist), (x2+25, y2-25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return dist

    def loop(self):
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.listLm.clear()

            ret, image = self.cap.read()
            image = cv2.flip(image, -1)
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.detector.process(image=imageRGB)

            self.findLandMarks(image, result)
            dist = self.findDist(image, 8, 4)

            cv2.imshow('Dino', image)

a = HandDetector()
a.loop()
