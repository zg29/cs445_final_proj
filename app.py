import cv2
import dlib
import numpy as np
import os
import csv
import sys
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, send_from_directory, url_for
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

def get_face_embedding_from_image(img_array):
    if img_array is None or img_array.size == 0:
        print("Error: Input image to get_face_embedding_from_image is empty or None.")
        return None
    try:
        # Ensure image is in BGR format if it came from cv2.imread or similar
        # If it's grayscale already, this won't hurt. If it has alpha, convert.
        if img_array.ndim == 3 and img_array.shape[2] == 4: # RGBA or BGRA
             img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR) # Or COLOR_RGBA2BGR

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"OpenCV error during color conversion in get_face_embedding_from_image: {e}")
        return None

    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected in the provided image region for Dlib.")
        return None

    face = faces[0]
    try:
        shape = sp(gray, face)
        # Important: compute_face_descriptor needs the original color image (or at least BGR)
        embedding = facerec.compute_face_descriptor(img_array, shape)
        return np.array(embedding)
    except Exception as e:
        print(f"Error computing face descriptor with Dlib: {e}")
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
    if not user_image_filename: return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    user_cropped_image = get_cropped_image(filepath, request.form, read_mode_cv2=cv2.IMREAD_COLOR)
    if user_cropped_image is None or user_cropped_image.size == 0:
        return "Error processing user image or invalid crop."

    # SSD models expect a 120x128 region after their original resize logic
    target_width = 120
    target_height = 128
    try:
        to_match = cv2.resize(user_cropped_image, (target_width, target_height))
    except cv2.error as e:
        return f"Error resizing cropped image for SSD: {e}. Cropped size: {user_cropped_image.shape}"


    min_ssd = sys.maxsize
    min_player_path = ""
    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Database images use the hardcoded crop
        img = cv2.resize(img, (260,190))[2:130,70:190] # Keep this consistent for DB images
        
        if img.shape != to_match.shape: # Ensure shapes match before SSD
            # This might happen if db image processing fails or is different
            # print(f"Shape mismatch: DB {img.shape}, User {to_match.shape} for {img_file}")
            continue 
        
        ssd = np.sum((to_match.astype(np.float32) - img.astype(np.float32)) ** 2) # astype for safety
        if (ssd < min_ssd):
            min_player_path = img_path
            min_ssd = ssd

    if not min_player_path:
        return "Could not find a match (no suitable DB images or all failed)."

    min_player_file = os.path.basename(min_player_path)
    min_player_id = min_player_file.split('.')[0]
    player_name = get_player_name(min_player_id)
    # Pass the original uploaded file path for display, not the crop.
    user_display_path = url_for('uploaded_file', filename=user_image_filename)
    player_display_path = url_for('static', filename=f'img/{min_player_file}')

    return render_template('result.html', title = "Color SSD",
                           similarity=f"{min_ssd:.2f}", user_image_path=user_display_path,
                           player_name=player_name, player_image_path=player_display_path)


@app.route('/ssd_gray', methods = ['POST'])
def ssd_gray_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename: return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    user_cropped_image = get_cropped_image(filepath, request.form, read_mode_cv2=cv2.IMREAD_GRAYSCALE)
    if user_cropped_image is None or user_cropped_image.size == 0:
        return "Error processing user image or invalid crop for grayscale SSD."

    target_width = 120
    target_height = 128
    try:
        to_match = cv2.resize(user_cropped_image, (target_width, target_height))
    except cv2.error as e:
         return f"Error resizing cropped image for Grayscale SSD: {e}. Cropped size: {user_cropped_image.shape}"


    min_ssd = sys.maxsize
    min_player_path = ""
    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (260,190))[2:130,70:190] # Keep this consistent for DB images
        
        if img.shape != to_match.shape:
            continue
        
        ssd = np.sum((to_match.astype(np.float32) - img.astype(np.float32)) ** 2)
        if (ssd < min_ssd):
            min_player_path = img_path
            min_ssd = ssd
            
    if not min_player_path:
        return "Could not find a match (no suitable DB images or all failed)."

    min_player_file = os.path.basename(min_player_path)
    min_player_id = min_player_file.split('.')[0]
    player_name = get_player_name(min_player_id)
    user_display_path = url_for('uploaded_file', filename=user_image_filename)
    player_display_path = url_for('static', filename=f'img/{min_player_file}')
    
    return render_template('result.html', title = "Grayscale SSD",
                           similarity=f"{min_ssd:.2f}", user_image_path=user_display_path,
                           player_name=player_name, player_image_path=player_display_path)

@app.route('/ncc_color', methods = ['POST'])
def ncc_color_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename: return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    user_cropped_image = get_cropped_image(filepath, request.form, read_mode_cv2=cv2.IMREAD_COLOR)
    if user_cropped_image is None or user_cropped_image.size == 0:
        return "Error processing user image or invalid crop for color NCC."

    target_width = 120
    target_height = 128
    try:
        to_match = cv2.resize(user_cropped_image, (target_width, target_height))
    except cv2.error as e:
        return f"Error resizing cropped image for Color NCC: {e}. Cropped size: {user_cropped_image.shape}"

    to_match_norms = []
    for c in range(to_match.shape[2] if to_match.ndim == 3 else 1): # Handle color/gray
        match_ch = to_match[:, :, c] if to_match.ndim == 3 else to_match
        mean_val = np.mean(match_ch)
        std_val = np.std(match_ch)
        to_match_norms.append((match_ch - mean_val) / (std_val + 1e-10))
    
    max_ncc = -sys.maxsize # NCC: Higher is better, so look for max
    best_player_path = ""

    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        db_img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Database images use the hardcoded crop
        db_img_cropped = cv2.resize(db_img_orig, (260,190))[2:130,70:190]
        
        if db_img_cropped.shape != to_match.shape:
            continue

        channel_nccs = []
        for c in range(db_img_cropped.shape[2] if db_img_cropped.ndim == 3 else 1):
            img_ch = db_img_cropped[:, :, c] if db_img_cropped.ndim == 3 else db_img_cropped
            mean_val = np.mean(img_ch)
            std_val = np.std(img_ch)
            img_ch_norm = (img_ch - mean_val) / (std_val + 1e-10)
            channel_nccs.append(np.mean(img_ch_norm * to_match_norms[c]))
        
        ncc = np.mean(channel_nccs)
        if (ncc > max_ncc): # NCC: higher is better
            best_player_path = img_path
            max_ncc = ncc
            
    if not best_player_path:
        return "Could not find a match for NCC Color."

    best_player_file = os.path.basename(best_player_path)
    player_id = best_player_file.split('.')[0]
    player_name = get_player_name(player_id)
    user_display_path = url_for('uploaded_file', filename=user_image_filename)
    player_display_path = url_for('static', filename=f'img/{best_player_file}')

    return render_template('result.html', title="Color NCC",
                           similarity=f"{max_ncc:.4f}", user_image_path=user_display_path,
                           player_name=player_name, player_image_path=player_display_path)


@app.route('/ncc_gray', methods = ['POST'])
def ncc_gray_model():
    user_image_filename = request.form.get('user_image_filename')
    if not user_image_filename: return "Invalid user filepath!"
    filepath = os.path.join(USER_UPLOADS_FOLDER, user_image_filename)

    user_cropped_image = get_cropped_image(filepath, request.form, read_mode_cv2=cv2.IMREAD_GRAYSCALE)
    if user_cropped_image is None or user_cropped_image.size == 0:
        return "Error processing user image or invalid crop for grayscale NCC."

    target_width = 120
    target_height = 128
    try:
        to_match = cv2.resize(user_cropped_image, (target_width, target_height))
    except cv2.error as e:
        return f"Error resizing cropped image for Grayscale NCC: {e}. Cropped size: {user_cropped_image.shape}"

    mean_to_match = np.mean(to_match)
    std_to_match = np.std(to_match)
    to_match_norm = (to_match - mean_to_match) / (std_to_match + 1e-10)
    
    max_ncc = -sys.maxsize # NCC: Higher is better
    best_player_path = ""

    for img_file in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, img_file)
        db_img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        db_img_cropped = cv2.resize(db_img_orig, (260,190))[2:130,70:190]

        if db_img_cropped.shape != to_match.shape:
            continue
            
        mean_db_img = np.mean(db_img_cropped)
        std_db_img = np.std(db_img_cropped)
        img_norm = (db_img_cropped - mean_db_img) / (std_db_img + 1e-10)
        
        ncc = np.mean(to_match_norm * img_norm)
        if (ncc > max_ncc): # NCC: higher is better
            best_player_path = img_path
            max_ncc = ncc
            
    if not best_player_path:
        return "Could not find a match for NCC Grayscale."

    best_player_file = os.path.basename(best_player_path)
    player_id = best_player_file.split('.')[0]
    player_name = get_player_name(player_id)
    user_display_path = url_for('uploaded_file', filename=user_image_filename)
    player_display_path = url_for('static', filename=f'img/{best_player_file}')
    
    return render_template('result.html', title="Grayscale NCC",
                           similarity=f"{max_ncc:.4f}", user_image_path=user_display_path,
                           player_name=player_name, player_image_path=player_display_path)

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
def get_cropped_image(filepath, request_form, read_mode_cv2=None, is_pil=False):
    crop_x_str = request_form.get('crop_x')
    crop_y_str = request_form.get('crop_y')
    crop_width_str = request_form.get('crop_width')
    crop_height_str = request_form.get('crop_height')
    preview_width_str = request_form.get('preview_width')
    preview_height_str = request_form.get('preview_height')

    if not all([crop_x_str, crop_y_str, crop_width_str, crop_height_str, preview_width_str, preview_height_str]):
        print("Warning: Crop coordinates not fully provided.")
        if is_pil:
            img = Image.open(filepath).convert("RGB")
            return img #
        else:
            img = cv2.imread(filepath, read_mode_cv2)
            return img 


    crop_x = float(crop_x_str)
    crop_y = float(crop_y_str)
    crop_width = float(crop_width_str)
    crop_height = float(crop_height_str)
    preview_width = float(preview_width_str)
    preview_height = float(preview_height_str)

    if preview_width == 0 or preview_height == 0: 
        return None 

    if is_pil:
        original_image = Image.open(filepath).convert("RGB")
        original_img_width, original_img_height = original_image.size
    else:
        original_image = cv2.imread(filepath, read_mode_cv2)
        if original_image is None: return None
        original_img_height, original_img_width = original_image.shape[:2]

    scale_w = original_img_width / preview_width
    scale_h = original_img_height / preview_height

    final_crop_x = int(crop_x * scale_w)
    final_crop_y = int(crop_y * scale_h)
    # For width and height, we scale the selected width/height directly
    final_crop_width = int(crop_width * scale_w)
    final_crop_height = int(crop_height * scale_h)
    
    # Ensure crop coordinates are within original image bounds
    final_crop_x = max(0, final_crop_x)
    final_crop_y = max(0, final_crop_y)
    
    if final_crop_x + final_crop_width > original_img_width:
        final_crop_width = original_img_width - final_crop_x
    if final_crop_y + final_crop_height > original_img_height:
        final_crop_height = original_img_height - final_crop_y

    if final_crop_width <= 0 or final_crop_height <= 0:
        print("Error: Invalid crop dimensions after scaling (<=0).")
        return None # Or return original image as fallback

    if is_pil:
        pil_crop_box = (
            final_crop_x,
            final_crop_y,
            final_crop_x + final_crop_width,
            final_crop_y + final_crop_height
        )
        return original_image.crop(pil_crop_box)
    else:
        return original_image[final_crop_y : final_crop_y + final_crop_height, final_crop_x : final_crop_x + final_crop_width]

if __name__ == "__main__":
    app.run(debug=True)
