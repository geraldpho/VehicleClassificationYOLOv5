from flask import Flask, request, jsonify, render_template, Response
import torch
from PIL import Image
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='yolov5/runs/train/exp29/weights/best.pt',
    force_reload=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(file.stream)
    results = model(img)

    # Convert results to JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # Render image with bounding boxes
    img_with_boxes = np.squeeze(results.render())
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Encode image as base64
    _, buffer = cv2.imencode('.jpg', img_with_boxes)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'detections': detections,
        'output_image': f"data:image/jpeg;base64,{img_base64}"
    })


def generate_frames():
    """Generate video frames for the live feed."""
    cap = cv2.VideoCapture(0)

    # Reduce frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # YOLO detection on each frame
        results = model(frame)
        annotated_frame = np.squeeze(results.render())

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield  frame as a response
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/live_feed')
def live_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)