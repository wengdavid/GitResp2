#1cv2
import cv2
#1引入mediapipe叫mp
import mediapipe as mp
#9 time
import time

#1读取VideoCapture放入cap
cap = cv2.VideoCapture(0)

#2手部模型函数放入mpHands
mpHands = mp.solutions.hands
#2呼叫mpHands函数放入hands()空可以
hands = mpHands.Hands(
              static_image_mode=False,
              model_complexity=0,
              min_detection_confidence=0.5,
              min_tracking_confidence=0.5)
#3把landmarks坐标画到手上mpDraw
mpDraw = mp.solutions.drawing_utils
#4设定连接点的颜色和粗度
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,0), thickness=5)
#4设定连接线的颜色和粗度
handConStyle = mpDraw.DrawingSpec(color=(0.255,255), thickness=10)
#9 ptime 和ctime
pTime = 0
cTime = 0

#1 while回圈读视频
while True:
    #1不断读取cap值，放入ret和img
   ret, img = cap.read()
   #1 如果读取ret
   if ret:
       #2将BGR转化成RGB并写入imgRGB
       imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       #2处理imgRGB结果放入result
       result = hands.process(imgRGB)
       #2打印landmarks坐标
       #print(result.multi_hand_landmarks)
       #6设置视窗的高度和宽度
       imgHeight = img.shape[0]
       imgWidth = img.shape[1]
       #3如果侦测到手
       if result.multi_hand_landmarks:
           #3把landmarks结果写入handLms
           for handLms in result.multi_hand_landmarks:
               #3在手上画出landmarks的点(1)，并用hand connections连接点(2)，并加入点和线handLmsStyle, handConStyle(4)
               mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
               #5用enumerate画出每一个手的数字(i=第几个点，lm=点的坐标)
               for i, lm in enumerate(handLms.landmark):
                   #5打印lm的坐标参数
                   #print(i, lm.x, lm.y)
                   #6设置点参数和视窗大小一致
                   xPos = int(lm.x * imgWidth)
                   yPos = int(lm.y * imgHeight)
                   #7在视频中打印点的数字，设置字体的样式，大小和颜色
                   cv2.putText(img, str(i),(xPos-25, yPos+5),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 0 ,0),2)
                   #8  4号大拇指画一个较大的⚪
                   if i == 4:
                       cv2.circle(img, (xPos, yPos), 15, (255,16,16), cv2.FILLED)
                   #6终端打印点的位置
                   print(i, xPos, yPos)

        #9设置一个刷新频率
       cTime = time.time()
       fps = 1/(cTime-pTime)
       pTime = cTime
       cv2.putText(img, f"FPS : {int(fps)}",(30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)

       #1显示叫img的视窗
       cv2.imshow('img',img)
    #1
   if cv2.waitKey(1)  == ord('q'):
    break
