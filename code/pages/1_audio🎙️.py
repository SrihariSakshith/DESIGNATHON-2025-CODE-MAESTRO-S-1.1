
import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
import os
import glob
import mediapipe as mp
import subprocess
import streamlit as st
import io
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
st.set_page_config(page_title='deepfake classification', layout='wide',initial_sidebar_state="collapsed")
model_name = "MelodyMachine/Deepfake-audio-detection-V2"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model1 = AutoModelForAudioClassification.from_pretrained(model_name)
def onchange():
    removefilesinfold("tempvid")
    removefilesinfold("temppics")
    removefilesinfold("tempaudio")
    removefilesinfold("output_of_photo")
def removefilesinfold(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)
def save_uploaded_file(uploaded_file):
    
    # Save the file temporarily
    save_path = os.path.join("tempaudio", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    return save_path
def classify_audio(path):
    
    audio, sr = librosa.load(path, sr=16000) 
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

   
    with torch.no_grad():
        outputs = model1(**inputs)

    
    predicted_label = torch.argmax(outputs.logits, dim=-1)





    confidences = F.softmax(outputs.logits, dim=1)
    percentages = confidences * 100
    print(percentages.tolist())
    return percentages.tolist()
def classify_recording(path):
   
    audio, sr = librosa.load(BytesIO(path), sr=16000) 

    
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model1(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=-1)





    confidences = F.softmax(outputs.logits, dim=1)
    percentages = confidences * 100
    print(percentages.tolist())
    return percentages.tolist()
st.write(" # $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ Audio:studio_microphone:")
with st.container(border= True):
    uploaded_file = st.file_uploader("Choose a audio file...",on_change=onchange)
col1,col2=st.columns(2)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    with col1:
        with st.container(border= True):
            st.audio(file_path)
            classify=st.button("classify")
    if(classify):
        audconf=classify_audio(file_path)
        with col2:
            with st.container(border= True):
                st.write(f"## Real percentage={float("{:.3f}".format(audconf[0][1]))}")
                st.progress(audconf[0][1]/100)
            with st.container(border= True):    
                st.write(f"## Fake percentage={float("{:.3f}".format(audconf[0][0]))}")
                st.progress(audconf[0][0]/100)

audio_bytes = audio_recorder(icon_size="1.5x")
if audio_bytes:
    with st.container(border= True):
        with col1:
            st.audio(audio_bytes)
            classify=st.button("classify",key=2)
    if(classify):

        x=classify_recording(audio_bytes)
        with col2:
            with st.container(border= True):
                st.write(f"# **Real percentage**={float("{:.2f}".format(x[0][1]))}")
                st.progress(x[0][1]/100)
            with st.container(border= True):
                st.write(f"# **Fake percentage**={float("{:.2f}".format(x[0][0]))}")
                st.progress(x[0][0]/100)
        

   

