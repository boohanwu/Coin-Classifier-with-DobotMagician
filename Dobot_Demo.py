import cv2 
import numpy as np
import time
import DobotDllType as dType

# Vision init
mask=None
capture=None
lastIndex = 5

# 吸盤中心點調整
X_Center = 320   
Y_Center = 240   

"""
<--負--Y--正-->

^
| 正
X
| 負
V
"""

# Load YOLO model
weights = "weights/yolov4-coin_last.weights"
cfg = "cfg/yolov4-coin.cfg"
classes = "classes/coin.names"

net = cv2.dnn.readNet(weights, cfg)

with open(classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

Video_num = 0   # 影像編號
state = "None"
capture = cv2.VideoCapture(Video_num)

# Dobot init
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

# Load Dll
api = dType.load()

# Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])

# 佇列釋放,工作執行函數
def work(lastIndex):
    # Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)
    # Wait for Executing Last Command 
    while lastIndex[0] > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)
    dType.SetQueuedCmdClear(api)
    
# 將相機讀取到的X,Y值,及要判斷的tag_id和吸盤高度輸入此函數
# 輸送帶便會將物件移至手臂下,並分類
def Dobot_work(cX, cY, tag_id, hei_z):
    # 以X_center,Y_center為中心,計算相機座標系統及手臂座標系統轉換.
    if(cY-Y_Center) >= 0:
        offy = (cY-Y_Center)*0.5001383    
    else:
        offy = (cY-Y_Center)*0.5043755    

    if(cX-X_Center) >= 0:
        offx = (X_Center-cX)*0.4921233      
    else:
        offx = (X_Center-cX)*0.5138767 
    obj_x = 210 + offx
    obj_y = offy

    # 輸送帶移動至手臂下
    dType.SetEMotor(api, 0, 1, 12500, 1)  
    dType.SetWAITCmd(api, 4850, isQueued=1)
    dType.SetEMotor(api, 0, 1, 0,1)
    dType.SetWAITCmd(api, 100, isQueued=1)
    # 手臂至影像計算後及座標轉換後obj_x,obj_y位置,吸取物件
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , 50, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , hei_z, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  1, isQueued=1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, obj_x, obj_y , 70, 0, 1)

    # 判斷是什麼物件並給予"各個類別"放置X,Y位置
    if(tag_id == 0 or tag_id == 1):     # 1元硬幣
        goal_x = 10
        goal_y = 210
        print("正在夾取1元")
    elif(tag_id == 2 or tag_id == 3):   # 5元硬幣
        goal_x = 50
        goal_y = 210
        print("正在夾取5元")
    elif(tag_id == 4 or tag_id == 5):   # 10元硬幣
        goal_x = 90
        goal_y = 210
        print("正在夾取10元")
    elif(tag_id == 6 or tag_id == 7):   # 50元硬幣
        goal_x = 130
        goal_y = 210
        print("正在夾取50元")

    # 依類別不同,將物件放置在各個位置.
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 40, 0, 1)
    dType.SetEndEffectorSuctionCup(api, 1,  0, isQueued=1)
    # 手爪控制函數說明
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, goal_x, -goal_y , 70, 0, 1)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 270, 0 , 50, 0, 1)
    lastIndex = dType.SetWAITCmd(api, 100, isQueued=1)
    work(lastIndex)
    print("End")

# Main start
if (state == dType.DobotConnect.DobotConnect_NoError):
    # Clean Command Queued
    dType.SetQueuedCmdClear(api)
    dType.SetPTPJointParams(api,200,200,200,200,200,200,200,200, isQueued = 1)
    dType.SetPTPCoordinateParams(api,200,200,200,200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)
    lastIndex = dType.SetWAITCmd(api, 2000, isQueued=1)
    work(lastIndex)

while(True):
    ret, img = capture.read()
    height, width, channels = img.shape     # default-> height:480, width:640
    window_scale = 1

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    #篩選預測框及分類閥值，輸出層有三個，外迴圈跑三次
    #內迴圈次數依據預測框的數量
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #第二次篩選預測框，參數依序為：預測框參數、存在物件信心度、信心度閥值及IoU閥值
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y-5), font, 2, color, 3)
            print("Class ID: %d, Object: %s, Confidence: %.2f, Center of X: %d, Center of Y: %d" % (class_ids[i], label, round(confidences[i], 2), center_x, center_y))
            
            if not ret:
                print("Failed to grab frame!")
                break

            keypress = cv2.waitKey(1)
            if keypress % 256 == 32:  # SPACE pressed
                img_name = "screenshot.jpg"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                screenshot = cv2.imread("screenshot.jpg")
                cv2.imshow("Screenshot", screenshot)
                # flag_start_work = True
                print (center_x, center_y, class_ids[i])
                Dobot_work(center_x, center_y, class_ids[i], 8)
                print("GO Work")

    frame = cv2.resize(img, (640 * window_scale, 480 * window_scale), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", frame)

#Stop to Execute Command Queued
dType.SetQueuedCmdStopExec(api)
#ser.close()
cv2.destroyAllWindows()
#Disconnect Dobot
dType.DisconnectDobot(api)
