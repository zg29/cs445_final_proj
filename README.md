# cs445_final_proj

Which NBA player do you look like?
We have used four different methods for this project to determine which NBA player you look most alike:
1. (Grayscale) Sum of Squared Differences (SSD)
2. (Grayscale) Normalized Cross Correlation (NCC)
3. Facebook AI Similarity Search (FAISS)
4. Facial Embedding Comparison using Dlib

## Instructions
Launch the project by executing app.py, then navigate to http://127.0.0.1:5000/ in a browser. 

After uploading an image on the homepage, you'll see an "Image Preview" screen like this:
<img width="687" alt="Screenshot 2025-05-09 at 5 07 41 PM" src="https://github.com/user-attachments/assets/b20833a2-49bf-4458-a979-bf3d3c24dc18" />

The buttons linking to different models are listed below (note FAISS and Dlib will take a minute!).

For the best SSD/NCC results, resize the image to a 260:190 aspect ratio. Note that FAISS and Dlib account for different facial positions.
