from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from flask import jsonify

import os
import cv2
import numpy as np


# model = YOLO(r"F:\IISc bnglr\yolov8n.pt")
app = Flask(__name__)

# Initialize the YOLO model
model_path = os.path.abspath("F:/IISc bnglr/yolov8n.pt")
model = YOLO(model_path)

# Ensure the 'uploads' and 'static/predicted' directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('static/predicted'):
    os.makedirs('static/predicted')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Save the uploaded image
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)

            # Perform object detection on the uploaded image
            frame = cv2.imread(image_path)  # Read the image
            results = model.predict(frame, save=True, imgsz=640, conf=0.5)
            print("Total Classes:", len(results.names))  # Print the results obtained from object detection

            # Define font parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White color
            thickness = 3
            line_type = cv2.LINE_AA

            # Iterate through the results and draw bounding boxes with class names
            for result in results:
                boxes = result.boxes.xyxy
                names = result.names  # Get class names directly

                for box, class_name in zip(boxes, names.values()):  # Iterate over boxes and class names
                    # Extract coordinates
                    x1, y1, x2, y2 = box

                    # Draw rectangle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)

                    # Add class name text
                    cv2.putText(frame, class_name, (int(x1), int(y1) - 10), font, font_scale, color, thickness, line_type)

            # Save the modified image with bounding boxes and class names
            predicted_image_path = os.path.join('static', 'predicted', image_file.filename)
            cv2.imwrite(predicted_image_path, frame)

            # Check if the image was saved successfully
            if os.path.exists(predicted_image_path):
                # Get the URL for the predicted image
                predicted_image_url = url_for('static', filename=os.path.join('predicted', image_file.filename))
                # Return the predicted image URL
                return jsonify({'predicted_image_url': predicted_image_url})
            else:
                return jsonify({'error_message': 'Failed to save the predicted image.'})
    return jsonify({'error_message': 'Prediction failed.'})




if __name__=='__main__':
    app.run(debug=True)
