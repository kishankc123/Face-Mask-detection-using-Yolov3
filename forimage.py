from datetime import datetime
import cv2
import numpy as np
import pytz

IST = pytz.timezone('Asia/Kathmandu')
net = cv2.dnn.readNet('yolov3_training_last_new.weights', 'yolov3.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.imread('test.jfif',1)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

height, width, _ = cap.shape

blob = cv2.dnn.blobFromImage(cap, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/4)
            y = int(center_y - h/4)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    # Add top-border to frame to display stats
    border_size = 100
    border_text_color = [255, 255, 255]
    cap = cv2.copyMakeBorder(cap, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)
    # calculate count values
    filtered_classids = np.take(class_ids, indexes)
    mask_count = (filtered_classids == 0).sum()
    nomask_count = (filtered_classids == 1).sum()

    # display count
    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(cap, text, (0, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
    # display status
    text = "Status:"
    cv2.putText(cap, text, (width - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
    ratio = nomask_count / (mask_count + nomask_count + 0.000001)

    if ratio >= 0.1 and nomask_count >= 3:
        text = "Danger !"
        cv2.putText(cap, text, (width - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, [26, 13, 247], 2)
        msg = "**Face Mask System Alert** \n\n"
        msg += "Status: \n" + str(text)+"\n"
        msg += "No_Mask Count: " + str(nomask_count) + " \n"
        msg += "Mask Count: " + str(mask_count) + " \n"
        datetime_ist = datetime.now(IST)
        msg += "Date-Time of alert: \n" + datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')

    elif ratio != 0 and np.isnan(ratio) != True:
        text = "Warning !"
        cv2.putText(cap, text, (width - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, [0, 255, 255], 2)

    else:
        text = "Safe "
        cv2.putText(cap, text, (width - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, [0, 255, 0], 2)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(cap, (x,y), (x+w, y+h), color, 2)
            cv2.putText(cap, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('FaceMask Detection', cap)
    key = cv2.waitKey(0)
    if key==27:
        break

cv2.destroyAllWindows()