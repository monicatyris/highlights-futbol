from moviepy.editor import *
import streamlit as st
import numpy as np
import pandas as pd
import os


def create_clips(frames, vid, num_clips, event_str):
    n = 1
    f = frames[0]
    if len(f) == 0:
        st.write("No events")
    else:
        for j in range(0, len(f)):
            diff_values = np.ediff1d(f)
            diff = np.append(diff_values, f[-1] + 121)
            # print("j {} f[j]{}".format(j, f[j]))
            # print("Diff {}".format(diff))
            # print("diff[j] {} diff[j] > 120 {}".format(diff[j], diff[j] > 120))
            # print("round(f[j] / 120, 2) - 20 {} round(f[j] / 120, 2) - 20 > 0 {}".format(round(f[j] / 120, 2) - 20, round(f[j] / 120, 2) - 20 > 0))
            if diff[j] > 120 and round(f[j] / 120, 2)*60 - 20 > 0:
                t1 = round((f[j] / 120), 2)*60 -20 # 20s before
                t2 = round((f[j] / 120), 2)*60 +20  # 20s after
                # print("n {} n <= num_clips {}".format(n, n <= num_clips))

                if n <= num_clips:
                    filename = "clip_" + str(event_str) + str(n) + ".mp4"
                    newclip = VideoFileClip(vid).subclip(t1, t2)
                    newclip.write_videofile(filename, fps=newclip.fps)
                    st.video(filename, 'video/mp4')
                    n += 1

def save_clips(frames, vid, num_clips, event_str):

    vid_name = os.path.splitext(vid)[0]
    newpath = os.path.join(vid_name, event_str)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    n = 1
    f = frames[0]
    df = pd.DataFrame(columns=['game_name', 'clip', 'timestamp', 'event'])
    if len(f) == 0:
        st.write("No events")
    else:
        for j in range(0, len(f)):
            diff_values = np.ediff1d(f)
            diff = np.append(diff_values, f[-1] + 121)
            if diff[j] > 120 and round(f[j] / 120, 2)*60 -20 > 0:
                t1 = round(f[j] / 120, 2)*60 -20 # 20s before
                t2 = round(f[j] / 120, 2)*60 +20  # 20s after
                if n <= num_clips:
                    filename = os.path.join(newpath, "clip_" + str(event_str) + str(n) + ".mp4")
                    newclip = VideoFileClip(vid).subclip(t1, t2)
                    newclip.write_videofile(filename, fps=newclip.fps)
                    df = df.append({'game_name': vid_name, 'clip': "clip_" + str(event_str) + str(n) + ".mp4", 'timestamp': round(f[j] / 120, 2)*60, 'event': str(event_str)}, ignore_index=True)
                    st.video(filename, 'video/mp4')
                    n += 1
    
    if not df.empty:
        df.to_csv(os.path.join(newpath, "record.csv"), index=False)