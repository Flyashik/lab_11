import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image,ImageOps
import pickle
import pandas as pd
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = pd.read_csv('data.csv')    
with open("model.pkl", "rb") as f:
    neighbors = pickle.load(f)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(image)
    img = img.save("img.jpg")
    image = preprocess(Image.open("img.jpg")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        arr = image_features[0].numpy()
    for id in (neighbors.kneighbors([arr])[1][0]):
        path = data[data['id']==id]['path']
        image = Image.open(path.astype('string').values[0])
        st.image(image)

