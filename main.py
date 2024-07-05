from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

model = YOLO('yolov8s.pt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    detections = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        detections.append({
            "class": model.names[int(cls)],
            "confidence": float(conf),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)
