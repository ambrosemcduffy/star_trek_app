from flask import Flask, request, render_template, send_from_directory
import os
import torch
import cv2
from torchvision import transforms
from model import vgg16_pretrain

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names
with open("data/class_names.txt", "r") as f:
    names_l = f.readlines()

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

vgg16_model = vgg16_pretrain()
vgg16_model.load_state_dict(torch.load("model/_strek_model_save2.pt", map_location=device))
vgg16_model.to(device)
vgg16_model.eval()

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image, (224, 224))
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean, std)(img)
    img = img.reshape(1, 3, 224, 224).to(device)

    with torch.no_grad():
        pred = vgg16_model(img)
        pred = torch.argmax(torch.exp(pred), axis=1)
        name = names_l[pred.detach().cpu().numpy()[0]].strip()

    return render_template("result.html", name=name, filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    import socket

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n🚀 App running! Open this on your phone:\nhttp://{local_ip}:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=True)