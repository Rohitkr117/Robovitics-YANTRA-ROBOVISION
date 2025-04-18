import mediapipe as mp
import cv2
import time

video = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1)
mpDraw = mp.solutions.drawing_utils

tip = [4, 8, 12, 16, 20]
fingerName = ['Thumb', 'Index_Finger', 'Middle_Finger', 'Ring_Finger', 'Little_finger']

prevTime = 0

while True:
    sucess, img = video.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmlist = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmlist.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # print(lmlist)
    if(len(lmlist)):
        finger = []
        if((lmlist[tip[0]][1] > lmlist[tip[0]-1][1] and lmlist[tip[0]][2] >= lmlist[tip[0]-1][2]) or (lmlist[tip[0]][1] <  lmlist[tip[0]-1][1] and lmlist[tip[0]][2] <= lmlist[tip[0] -1][2])):
            finger.append(1)

        else:
            finger.append(0)
        
        for id in range(1, 5):
            if lmlist[tip[id]][2] < lmlist[tip[id] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)

        totalFingers = sum(finger)

        if(totalFingers == 1):
            for fin in range(5):
                if finger[fin] == 1:
                    upFinger = fingerName[fin]
                    cv2.putText(img, str(upFinger), (275,445), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0), 3)
        elif totalFingers == 5:
            cv2.putText(img, "Palm", (275,445), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0), 3)
        elif totalFingers == 0:
            cv2.putText(img, "Fist", (275,445), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0), 3)

        cv2.putText(img, str(totalFingers), (575,45), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0), 3)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime           

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()