import numpy as np
import streamlit as st
import model


def spotting_actions(feat_pca_name):
    chunksize = 120
    framerate = 2
    feat_pca = np.load(feat_pca_name, allow_pickle=True)
    with st.spinner('Loading model...'):
        network = model.define_sppoting_model(chunksize, framerate, feat_pca)
    # Load the network and its weights
    with st.spinner('Loading weights...'):
        network.load_weights('best_average_mAP.h5')

    with st.spinner('Detecting highlights...'):

        features = list()

        # Get the features and labels from the set
        features = np.load(feat_pca_name, allow_pickle=True)

        # Compute the chunk size and receptive field from the arguments
        receptivefield = 40
        chunk_size = chunksize*framerate
        receptive_field = int(receptivefield*framerate/2)
        # For the all videos of the testset, get the graphs and numpy arrays of the results
        counter = 0
        # Arrays for saving the segmentations, the detections and the detections scores
        prediction = np.zeros((features.shape[0],3))
        timestamps_scores = np.zeros((features.shape[0],3))-1

        # Preprocessing of the feature data
        data_expanded = np.expand_dims(features, axis=0)
        if data_expanded.shape[-1] != 3:
            data_expanded = np.expand_dims(data_expanded, axis=-1)

        # Loop over the entire game, chunk by chunk
        start = 0
        last = False
        while True:

            # Get the output for that chunk and retrive the segmentations and the detections
            tmp_output = network.predict_on_batch(data_expanded[:,start:start+chunk_size])
            output = tmp_output[0][0]
            timestamps = tmp_output[1][0]
            # Store the detections confidence score for the chunck
            timestamps_long_score = np.zeros(output.shape)-1
            # Store the detections and confidence scores in a chunck size array
            for i in np.arange(timestamps.shape[0]):
                timestamps_long_score[ int(np.floor(timestamps[i,1]*(output.shape[0]-1))) , int(np.argmax(timestamps[i,2:])) ] = timestamps[i,0]

            # For the first chunk
            if start == 0:
                prediction[0:chunk_size-receptive_field] = output[0:chunk_size-receptive_field]
                timestamps_scores[0:chunk_size-receptive_field] = timestamps_long_score[0:chunk_size-receptive_field]
            # For the last chunk
            elif last:
                prediction[start+receptive_field:start+chunk_size] = output[receptive_field:]
                timestamps_scores[start+receptive_field:start+chunk_size] = timestamps_long_score[receptive_field:]
                break
            # For every other chunk
            else:
                prediction[start+receptive_field:start+chunk_size-receptive_field] = output[receptive_field:chunk_size-receptive_field]
                timestamps_scores[start+receptive_field:start+chunk_size-receptive_field] = timestamps_long_score[receptive_field:chunk_size-receptive_field]

            start += chunk_size - 2 * receptive_field
            if start + chunk_size >= features.shape[0]:
                start = features.shape[0] - chunk_size
                last = True

        timestamps_final = np.where(timestamps_scores >= 0.34, 1.35, 0)

    return 1-prediction, timestamps_final, timestamps_scores