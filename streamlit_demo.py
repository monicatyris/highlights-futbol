"""
Usage:

streamlit run streamlit_demo.py --server.maxUploadSize=2056

"""

import numpy as np
import streamlit as st
import tempfile
import cv2
import os

import preprocessing
import spotting
import create_timeline
import cutting_clips
import create_record


def file_selector(folder_path='.'):
    filenames = ['features_1half_Juventus_Udinese.npy', 'features_1half_Liverpool_RM.npy']
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


if __name__ == '__main__':

    st.header("Highlights Extraction Module")
    st.write("Choose a video, get highlights and summary clip")
    uploaded_file = st.sidebar.file_uploader("Choose a video...")
    page_names = ["Provide Features", "Preprocess Video"]

    if uploaded_file is not None:
        vid = uploaded_file.name
        st.sidebar.success('Video uploaded!')

        vf = cv2.VideoCapture(vid)
        page = st.sidebar.radio("Please, indicate what you would like to do next", page_names)

        if page == "Provide Features":
            feat_reduced = file_selector()
            st.sidebar.success('Features uploaded!')
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            ResNETmodel = preprocessing.create_model()
            features = preprocessing.extract_features(video_path, 2, "crop", None, None, ResNETmodel)
            feat_reduced = preprocessing.compute_pca(features)
            st.sidebar.success('Features extracted!')


        segmentations, ts_f, detections = spotting.spotting_actions(feat_reduced)

        show = st.sidebar.checkbox('Display highlights graphic')
        if show:
            create_timeline.show_detections(segmentations, detections)

        goal_frames = np.where(ts_f[:, 0] > 0)
        card_frames = np.where(ts_f[:, 1] > 0)
        subs_frames = np.where(ts_f[:, 2] > 0)

        if st.button('Create clips with Goal highlights'):
            st.write('Goal highlights:')
            cutting_clips.save_clips(goal_frames, vid, 3, "goal")
        if st.button('Create clips with Cards highlights'):
            st.write('Cards highlights:')
            cutting_clips.save_clips(card_frames, vid, 3, "card")
        if st.button('Create clips with Substitutions highlights'):
            st.write('Substitution highlights:')
            cutting_clips.save_clips(subs_frames, vid, 3, "subs")
        if st.button('Create timestamps record'):
            st.write('Timestamps record:')
            st.dataframe(create_record.join_csv(vid))




