import torch
from torchvision import transforms
from torch import nn
from PIL import Image
from d2l import torch as d2l
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

model = get_net()
model.load_state_dict(torch.load("./model"))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
def load_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Perform inference
def classify_image(image_bytes):
    image_tensor = load_image(image_bytes)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    try:
        img_bytes = file.read()
        predicted_class = classify_image(img_bytes)
        return jsonify({'predicted_class': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
