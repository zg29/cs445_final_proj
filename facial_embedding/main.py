# import cv2
import dlib
import numpy as np
import os
import csv
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from PIL import Image

# create detector
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def get_face_embedding(image_path):
    img = Image.open(image_path)
    img_gray = img.convert("L")
    gray = np.array(img_gray)
    gray_rgb = np.stack([gray] * 3, axis=-1)

    faces = detector(gray)
    
    if len(faces) == 0:
        return None 

    face = faces[0]
    shape = sp(gray, face)
    embedding = facerec.compute_face_descriptor(gray_rgb, shape)
    
    return np.array(embedding)

# perform dot product to get a value of similarity
def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None

    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# find name associated with player id
def get_player_name(player_id, csv_file='players.csv'):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['playerid'] == str(player_id):
                return f"{row['fname']} {row['lname']}"
    return None

def display_images_side_by_side(image1_path, image2_path, player_name):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  
    plt.imshow(img1)
    plt.title("User Image")
    plt.axis('off')
    plt.subplot(1, 2, 2) 
    plt.imshow(img2)
    plt.title(f"{player_name}")
    plt.axis('off')

    plt.show()


def main():
    user_image_path = input("Enter the path to the image you want to compare: ")

    user_embedding = get_face_embedding(user_image_path)
    img_folder = 'static/img'
    most_similar_image = None
    highest_similarity = 0
    most_similar_player_id = None

    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)

        if img_file.startswith('.') or ':zone.identifier' in img_file:
            print(f"Skipping file: {img_file} (zone identifier or hidden)")
            continue

        # only accept .png, .jpg, .jpeg
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_embedding = get_face_embedding(img_path)

        similarity = compare_faces(user_embedding, img_embedding)
        if similarity is not None and similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_image = img_file
            
            player_id = int(img_file.split('.')[0])
            most_similar_player_id = player_id

    if most_similar_image:
        print(f"The most similar image to {user_image_path} is {most_similar_image} with a similarity score of {highest_similarity:.4f}")
        
        # get the player's name based on the player ID
        player_name = get_player_name(most_similar_player_id)
        if player_name:
            print(f"The player associated with this image is: {player_name}")
        else:
            print("Player ID not found in the CSV.")

        # display the most similar player's image
        player_image_path = os.path.join(img_folder, most_similar_image)
        display_images_side_by_side(user_image_path,player_image_path, player_name)
    else:
        print("No similar faces found in the folder.")

if __name__ == "__main__":
    main()