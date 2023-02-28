from SoccerNet.DataLoader import FrameCV  # pip install Soccernet
from tensorflow import keras
from tensorflow.keras.models import Model  # pip install tensorflow (==2.3.0)
from tensorflow.keras.applications.resnet import preprocess_input

import streamlit as st
import pickle as pkl
import os

# GPU could be 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


def create_model():
    # create pretrained encoder (here ResNet152, pre-trained on ImageNet)
    base_model = keras.applications.resnet.ResNet152(include_top=True,
                                                     weights='imagenet',
                                                     input_tensor=None,
                                                     input_shape=None,
                                                     pooling=None,
                                                     classes=1000)

    # define model with output after polling layer (dim=2048)
    m = Model(base_model.input,
              outputs=[base_model.get_layer("avg_pool").output])
    m.trainable = False
    return m


def extract_features(vid_path, fps, transform, start, duration, mod):
    with st.spinner('Extracting features...'):
        videoLoader = FrameCV(vid_path, fps, transform, start, duration)
        frames = preprocess_input(videoLoader.frames)
    st.success('Features extracted!')
    return mod.predict(frames, batch_size=64, verbose=1)


def compute_pca(feat):
    with st.spinner('Computing PCA...'):
        with open('pca_512_TF2.pkl', "rb") as fobj:
            pca = pkl.load(fobj)
        with open('average_512_TF2.pkl', "rb") as fobj:
            average = pkl.load(fobj)
        feat = feat - average
    st.success('PCA computed!')
    return pca.transform(feat)
