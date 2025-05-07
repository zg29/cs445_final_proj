import cv2
import dlib
import numpy as np
import os
import csv
import sys
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import faiss
from PIL import Image

app = Flask(__name__)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

IMG_FOLDER = 'static/img'
USER_UPLOADS_FOLDER = 'static/user_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
resnet.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None

    face = faces[0]
    shape = sp(gray, face)
    embedding = facerec.compute_face_descriptor(img, shape)
    return np.array(embedding)

def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    return 1 - cosine(embedding1, embedding2)

def get_player_name(player_id, csv_file='players.csv'):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['playerid'] == str(player_id):
                return f"{row['fname']} {row['lname']}"
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image_file(img_file):
    return not img_file.startswith('.') and ':zone.identifier' not in img_file

def extract_feature(img):
    tensor = preprocess(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        features = resnet(tensor).squeeze().numpy()  # shape: (2048,)
    return features / np.linalg.norm(features)  # normalize

def init_faiss():
    features = []
    player_images = os.listdir(IMG_FOLDER)
    for img_file in player_images:
        img_path = os.path.join(IMG_FOLDER, img_file)
        img = Image.open(img_path).convert("RGB").crop((70, 2, 190, 130))
        features.append(extract_feature(img))
    features = np.array(features)
    faiss_index = faiss.IndexFlatL2(features.shape[1])
    faiss_index.add(features)
    return faiss_index

def extract_feature_faiss(img):
    tensor = preprocess(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        features = resnet(tensor).squeeze().numpy()  # shape: (2048,)
    return features / np.linalg.norm(features)  # normalize

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_image_path = os.path.join(USER_UPLOADS_FOLDER, filename)
        file.save(user_image_path)
        return render_template('preview.html', user_image_filename=filename)

    return "Invalid file"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(USER_UPLOADS_FOLDER, filename)

@app.route('/ssd_color', methods = ['POST'])
def ssd_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename:
        return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    to_match = cv2.imread(filepath, cv2.IMREAD_COLOR)
    to_match = cv2.resize(to_match, (260, 190))[2:130,70:190]
    min_ssd = sys.maxsize
    min_player_path = ""
    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img[2:130,70:190]
        ssd = np.sum((to_match - img) ** 2)
        if (ssd < min_ssd):
            min_player_path = img_path
            min_ssd = ssd

    min_player_file = os.path.basename(min_player_path)
    min_player_id = min_player_file.split('.')[0]
    player_name = get_player_name(min_player_id)
    return render_template('result.html', title = "SSD",
                            similarity=min_ssd, user_image_path=filepath,
                            player_name=player_name, player_image_path=min_player_path)

@app.route('/ssd_gray', methods = ['POST'])
def ssd_gray_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename:
        return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    to_match = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    to_match = cv2.resize(to_match, (260, 190))[2:130,70:190]
    min_ssd = sys.maxsize
    min_player_path = ""
    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[2:130,70:190]
        ssd = np.sum((to_match - img) ** 2)
        if (ssd < min_ssd):
            min_player_path = img_path
            min_ssd = ssd

    min_player_file = os.path.basename(min_player_path)
    min_player_id = min_player_file.split('.')[0]
    player_name = get_player_name(min_player_id)
    return render_template('result.html', title = "Grayscale SSD",
                            similarity=min_ssd, user_image_path=filepath,
                            player_name=player_name, player_image_path=min_player_path)

@app.route('/faiss', methods = ['POST'])
def faiss_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename:
        return "Invalid user filepath!"

    print("Initializing FAISS model, this may take a minute...")
    faiss_index = init_faiss()
    print("Done initializing FAISS model!")
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)
    to_match = Image.open(filepath).convert("RGB").resize((260, 190)).crop((70, 2, 190, 130))

    to_match_faiss = extract_feature(to_match).reshape(1, -1)
    sim, idx = faiss_index.search(to_match_faiss, 1)
    player_file = os.listdir(IMG_FOLDER)[idx[0][0]]
    player_name = get_player_name(player_file.split(".")[0])
    best_match_path = os.path.join(IMG_FOLDER, player_file)
    return render_template('result.html', title = "FAISS",
                            similarity=sim[0][0], user_image_path=filepath,
                            player_name=player_name, player_image_path=best_match_path)

@app.route('/dlib', methods = ['POST'])
def dlib_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename:
        return "Invalid user filepath!"
    
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)
    print("Initializing Dlib embedding, this may take a minute...")
    user_embedding = get_face_embedding(filepath)
    if user_embedding is None:
        return "Could not extract embedding from the user's image."
    print("Done initializing Dlib embedding!")
    most_similar_image = None
    highest_similarity = 0
    most_similar_player_id = None
    
    for img_file in os.listdir(IMG_FOLDER):
        if not is_valid_image_file(img_file):
            continue

        img_path = os.path.join(IMG_FOLDER, img_file)
        img_embedding = get_face_embedding(img_path)
        if img_embedding is not None:
            similarity = compare_faces(user_embedding, img_embedding)
            if similarity is not None and similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_image = img_file
                player_id = int(img_file.split('.')[0])
                most_similar_player_id = player_id
    
    if most_similar_image:
        player_name = get_player_name(most_similar_player_id)
        player_image_path = os.path.join(IMG_FOLDER, most_similar_image)
        return render_template('result.html', title = "Dlib",
                                similarity=highest_similarity, user_image_path=filepath,
                                player_name=player_name, player_image_path=player_image_path)

    else:
        return "No similar faces found."

if __name__ == "__main__":
    app.run(debug=True)
