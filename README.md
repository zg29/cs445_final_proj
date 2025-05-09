# cs445_final_proj

Which NBA player do you look like?
We have used four different methods for this project to determine which NBA player you look most alike:
1. (Grayscale) Sum of Squared Differences (SSD)
2. (Grayscale) Normalized Cross Correlation (NCC)
3. Facebook AI Similarity Search (FAISS)
4. Facial Embedding Comparison using Dlib

## Instructions

First, download [shape_predictor_68_face_landmarks.dat](https://uofi.app.box.com/file/1857984083786) and [dlib_face_recognition_resnet_model_v1.dat](https://uofi.app.box.com/file/1857995449609). Store these files into the models folder.

Launch the project by executing app.py, then navigate to http://127.0.0.1:5000/ in a browser. 

After uploading an image on the homepage, you'll see an "Image Preview" screen like this:
<img width="538" alt="Screenshot 2025-05-07 at 2 04 01 AM" src="https://github.com/user-attachments/assets/afb43fb5-3521-42d1-80b1-d005101444c7" />

The buttons linking to different models are listed below (note FAISS and Dlib will take a minute!).

For the best SSD/NCC results, resize the image to 260x190 and center the face at roughly the top-center. Note that FAISS and Dlib account for different facial positions.
