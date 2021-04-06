#import libs
import cv2 as cv
import numpy as np 


#
url = 'http://192.168.1.2:8080/video'
cap = cv.VideoCapture(url)
classfile = 'coco.names'
classnames = []
whT = 320
confthreshold = 0.5
nms_threshold = 0.3

with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

#models
modelconfig = 'yolov3.cfg'
modelweight = 'yolov3.weights'

net = cv.dnn.readNet(modelconfig,modelweight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def find_obj(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classids = []
    confi = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence > confthreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classids.append(classid)
                confi.append(float(confidence))

    print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confi,confthreshold,nms_threshold)

    for i in indices:
        i =i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2], box[3]
        cv.rectangle(img,(x-100,y),(x+w,y+h),(0,255,0),2)
        cv.putText(img,f'{classnames[classids[i]].upper()}{confi[i]*100}%',(x,y),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layernames = net.getLayerNames()

    outputnames= [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputnames)

    find_obj(outputs,img)
    cv.imshow('window',img)

    key = cv.waitKey(1)
    if key == ord('q') :
        cap.release()
        cv.destroyAllWindows()
        exit(1)