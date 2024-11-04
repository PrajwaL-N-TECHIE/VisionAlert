import cv2
import torch
import numpy as np
import os
from datetime import datetime
import pywhatkit as kit

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

if not os.path.exists('cropped_objects'):
    os.makedirs('cropped_objects')

cap = cv2.VideoCapture(0)

def send_whatsapp_notification(label, count):
    message = f"Detected {label}: {count} times."
    phone_number = "+919655095004"
    try:
        kit.sendwhatmsg_instantly(phone_number, message, wait_time=15, tab_close=True)
        print(f"WhatsApp Notification Sent: {message}")
    except Exception as e:
        print(f"Failed to send WhatsApp notification: {str(e)}")

def crop_and_save(image, bbox, label, count):
    x1, y1, x2, y2 = bbox
    cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"cropped_objects/{label}_{count}_{timestamp}.jpg"
    cv2.imwrite(filename, cropped_img)
    print(f"Saved cropped image: {filename}")

notified_objects = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    object_count = {}

    for idx, detection in detections.iterrows():
        confidence = detection['confidence']
        if confidence < 0.5:
            continue
        label = detection['name']
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        if label in object_count:
            object_count[label] += 1
        else:
            object_count[label] = 1
            if label not in notified_objects or notified_objects[label] < object_count[label]:
                send_whatsapp_notification(label, object_count[label])
                notified_objects[label] = object_count[label]

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        crop_and_save(frame, (x1, y1, x2, y2), label, object_count[label])

    y_offset = 30
    for obj, count in object_count.items():
        count_text = f"{obj}: {count}"
        cv2.putText(frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 20

    cv2.putText(frame, f"Timestamp: {timestamp}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("YOLOv5 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
