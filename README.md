# Traffic-Objects
<h1>YOLOv8 Object Detection Model Prediction Project</h1>
Introduction
This project demonstrates the use of the YOLOv8 (You Only Look Once version 8) object detection model for predicting objects in images. The project includes a Flask web application that allows users to upload images, perform object detection using the YOLOv8 model, and view the predicted results.

Features
Upload images for object detection.
Perform object detection using the YOLOv8 model.
View predicted objects overlaid on the uploaded image.
Ability to preview the predicted image before and after detection.
<h6>Technologies Used</h6>
<li>Python</li>
<li>Flask</li>
<li>YOLOv8 Model (Ultralytics)</li>
<li>OpenCV</li>
<h3>Object Detection Process:</h3>
<li>Upload Image: Users can upload an image containing various objects they want to detect. This image is then sent to the server for processing.</li>Upload Image: Users can upload an image containing various objects they want to detect. This image is then sent to the server for processing.

<li>YOLOv8 Model Prediction: Upon receiving the image, the server utilizes the YOLOv8 (You Only Look Once version 8) object detection model to predict the objects present in the image. YOLOv8 is a state-of-the-art deep learning model capable of detecting multiple objects in real-time with high accuracy.
</li>
<li>Bounding Box Visualization: The model detects objects within the image and draws bounding boxes around them to indicate their locations. Each bounding box is accompanied by a label indicating the class of the detected object (e.g., person, car, dog).
</li>
<li>Class Label Filtering: Optionally, unwanted or irrelevant class labels can be filtered out from the predictions to refine the results and focus only on specific objects of interest. This ensures that only relevant information is presented to the user.
</li>
<li>Display Predictions: The image with bounding boxes and class labels overlaid on it is displayed to the user, allowing them to visualize the detected objects within the context of the original image.
</li>
<h3>Key Features:</h3>
<li>Accuracy: The YOLOv8 model provides accurate predictions, enabling reliable detection of various objects within the image.</li>
<li>Real-time Processing: The model performs object detection efficiently, allowing for real-time processing of uploaded images.</li>
<li>Multiple Object Detection: The model is capable of detecting multiple objects belonging to different classes simultaneously within the same image.</li>
<li>Customization: Users have the flexibility to filter out specific class labels based on their preferences, tailoring the predictions to suit their needs.</li>
<h3>Use Cases:</h3>
<li>Security Surveillance: Detecting intruders, vehicles, or suspicious activities in security camera footage.</li>
<li> Autonomous Vehicles: Identifying pedestrians, other vehicles, and traffic signs to assist autonomous vehicles in navigation.</li>
<li>Retail Analytics: Counting and tracking customer movements, monitoring product inventory, and analyzing customer behavior in retail environments.</li>
<li>Environmental Monitoring: Detecting wildlife, vegetation, and other objects of interest in environmental monitoring applications.</li>
<h3> Refer this video: https://drive.google.com/file/d/170XNcVL6J-H7n7eMWiu7YplKKY83NbDf/view?usp=drivesdk</h3>
