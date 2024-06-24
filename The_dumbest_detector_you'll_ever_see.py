import cv2
import numpy as np
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load YOLO
net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg")
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load coco.names
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load sound
hello_sound = pygame.mixer.Sound("Air Horn Sound Effect.mp3")

# Initialize the camera
cap = cv2.VideoCapture(0)

def detect_people(frame):
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Showing informations on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    return len(indexes) > 0

last_detection_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detect_people(frame):
        if time.time() - last_detection_time > 5:  # Play sound at most once every 5 seconds
            hello_sound.play()
            last_detection_time = time.time()
        cv2.putText(frame, "HELLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
