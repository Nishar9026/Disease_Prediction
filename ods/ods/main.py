import cv2
thres=0.5 # Threshold to detect object
cap=cv2.VideoCapture(0) # Start video capture using your default camera
cap.set(3,648) #3. Width=648 px
cap.set(4,448) #4. Height=448 px
cap.set(10,70) #10. Brightness=70

#Class name Loading
className=[] #Empty List
classFile='coco.names'
with open(classFile,'rt') as f:
     className=f.read().rstrip('\n').split('\n')
configPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" #Features
weightPath="frozen_inference_graph.pb" #Model
net=cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320,320) #Resize Image to 320x320 pixels
net.setInputScale(1.0/127.5) #Normalize pixel values to[-1,1]
net.setInputMean((127.5,127.5,127.5)) #Mean substraction ,substract 127.5 from each channel
net.setInputSwapRB(True) #Swap channels from BGR to RGB

while True:
       success,img=cap.read()
       classIds, confs, bbox=net.detect(img,confThreshold=thres)
       if len(classIds)!=0:
             for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                  cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                  cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
       cv2.imshow("Output",img)
       if cv2.waitKey(1) & 0xFF==ord('q'):
             break