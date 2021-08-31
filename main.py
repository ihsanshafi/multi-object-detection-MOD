#import libs
import cv2 as cv
import numpy as np 


#variables
url = 'http://192.168.1.2:8080/video'

cap = cv.VideoCapture(0)
classfile = 'coco.names'
classnames = []
whT = 320
confthreshold = 0.4
nms_threshold = 0.3

"""with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
"""
classnames=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#models
modelconfig = 'yolov3.cfg'
modelweight = 'yolov3.weights'

colors = np.random.uniform(0,255, size=(len(classnames),3))

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

    #print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confi,confthreshold,nms_threshold)

    for i in indices:
        i =i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2], box[3]
        color = colors[i]
        cv.rectangle(img,(x-100,y),(x+w,y+h),color,2)
        text = f'{classnames[classids[i]].upper()}{int(confi[i]*100)}%'
        #text = "{}: {:.4f}%".format(classnames[classids[i]], round(confi[i]*100)).upper()
        cv.putText(img,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.6,color,2)

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
