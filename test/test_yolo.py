import cv2
import zmq
import base64
import json
from ultralytics import YOLO

model = YOLO("yolov8s-face-lindevs.pt")

ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://127.0.0.1:5555")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int).tolist()

    _, jpg = cv2.imencode(".jpg", frame)
    payload = {
        "image": base64.b64encode(jpg).decode(),
        "boxes": boxes
    }

    sock.send_json(payload)

    cv2.imshow("Python Camera", frame)
    if cv2.waitKey(1) == 27:
        break
