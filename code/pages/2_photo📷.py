import streamlit as st
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
from safetensors.torch import load_file
import os
import glob
warnings.filterwarnings("ignore")
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from transformers import AutoModelForImageClassification,pipeline
st.set_page_config(page_title='deepfake classification', layout='wide',initial_sidebar_state="collapsed")
model = AutoModelForImageClassification.from_pretrained("not-lain/deepfake", trust_remote_code=True)

pipe = pipeline(model="deepfake",trust_remote_code=True,device=DEVICE)
def onchange():
    removefilesinfold("tempvid")
    removefilesinfold("temppics")
    removefilesinfold("tempaudio")
    removefilesinfold("output_of_photo")
def removefilesinfold(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)
# mtcnn = MTCNN(
#     select_largest=False,
#     post_process=False,
#     device=DEVICE
# ).to(DEVICE).eval()
# model = InceptionResnetV1(
#     pretrained="vggface2",
#     classify=True,
#     num_classes=1,
#     device=DEVICE
# )

# checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(DEVICE)
# model.eval()
def predict(path):
    """Predict the label of the input_image"""
    # face = mtcnn(input_image)
    # if face is None:
    #     raise Exception('No face detected')
    # face = face.unsqueeze(0) # add the batch dimension
    # face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # # convert the face into a numpy array to be able to plot it
    # prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    # prev_face = prev_face.astype('uint8')

    # face = face.to(DEVICE)
    # face = face.to(torch.float32)
    # face = face / 255.0
    # face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    # target_layers=[model.block8.branch1[-1]]
    # use_cuda = True if torch.cuda.is_available() else False
    # cam = GradCAM(model=model, target_layers=target_layers)
    # targets = [ClassifierOutputTarget(0)]

    # grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    # face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # with torch.no_grad():
    #     output = torch.sigmoid(model(face).squeeze(0))
    #     prediction = "real" if output.item() < 0.5 else "fake"
        
    #     real_prediction = 1 - output.item()
    #     fake_prediction = output.item()
        
    #     confidences = {
    #         'real': real_prediction,
    #         'fake': fake_prediction
    #     }
    y=pipe.predict(path)
    cv2.imwrite("output_of_photo/out.png",y["face_with_mask"])
    

    return y["confidences"],"output_of_photo/out.png" 
def save_uploaded_file(uploaded_file):
    
    # Save the file temporarily
    save_path = os.path.join("temppics", uploaded_file.name)  # Change "uploads" to your desired folder
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    
    return save_path
st.write(" # $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ photo:camera:")
with st.container(border=True):
    uploaded_file = st.file_uploader("Choose a photo...",on_change=onchange)
col1,col2=st.columns(2)
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    with col1:
        with st.container(border=True):
            st.image(file_path)
            classify=st.button("classify")
    if(classify):
        
        conf,path=predict(file_path)
        with col2:
            with st.container(border=True):
                with st.container(border=True):
                    st.image(path)
                with st.container(border=True):
                    st.write(f"# Real percentage={float("{:.3f}".format(conf["real"]*100))}%")
                    st.progress(conf["real"])
                with st.container(border=True):
                    st.write(f"# Fake percentage={float("{:.3f}".format(conf["fake"]*100))}%")
                    st.progress(conf["fake"])
        