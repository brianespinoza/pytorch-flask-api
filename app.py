import io
import json
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
imagenet_class_index = json.load(open("imagenet_class_index.json"))

# Load pre-trained DenseNet model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.eval()

# Ensure image preprocessing matches ImageNet training
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# Get prediction and debug outputs
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)  # Get top 5 predictions

    print("Top 5 Predictions:")
    for i in range(5):
        idx = str(top5_idx[0, i].item())
        print(f"{imagenet_class_index[idx]} with probability {top5_prob[0, i].item()}")

    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())

    return imagenet_class_index[predicted_idx]

@app.route("/")
def index():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img_bytes = file.stream.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    return jsonify({"class_id": class_id, "class_name": class_name})

if __name__ == "__main__":
    app.run()
