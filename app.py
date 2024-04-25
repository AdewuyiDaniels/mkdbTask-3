from flask import Flask, request, jsonify
import cv2
import json

app = Flask(__name__)

# Load annotations from JSON files
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


@app.route('/detect_memory', methods=['POST'])
def detect_memory_api():
    # Get uploaded image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    
    # Determine whether the image contains memory or no-memory
    is_memory = request.form.get('is_memory')
    if is_memory is None:
        return jsonify({'error': 'Please specify whether the image contains memory or no-memory'}), 400
    
    # Load annotations based on memory or no-memory
    annotation_file = 'labels_memory_images_2024-04-25-05-23-12.json' if is_memory else 'labels_no_memory_images_2024-04-25-05-53-55.json'
    annotations = load_annotations(annotation_file)
    
    # Read the uploaded image
    image_data = image_file.read()
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Perform memory detection using YOLOv5
    memory_boxes = detect_memory(image_array, annotations)
    
    # Draw bounding boxes on the image
    for box in memory_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Convert image with bounding boxes to bytes
    _, encoded_image = cv2.imencode('.jpg', image_array)
    encoded_image_bytes = encoded_image.tobytes()
    
    return encoded_image_bytes

if __name__ == '__main__':
    app.run(debug=True)
