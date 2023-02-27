import pandas as pd
import streamlit as st
import os


def join_csv(vid):

    path = os.path.splitext(vid)[0]
    dir_list = os.listdir(path)
    df_append = pd.DataFrame(columns=['game_name', 'clip', 'timestamp', 'event'])
    
    for dir in dir_list:
        newpath = os.path.join(path, dir)
        csvpath = os.path.join(newpath, "record.csv")
        if os.path.isdir(newpath) and os.path.exists(csvpath):
            df_temp = pd.read_csv(csvpath)
            df_append = df_append.append(df_temp, ignore_index=True)

    if df_append.empty:
        st.write("No record info")
    df_append.to_csv(os.path.join(path, "record.csv"), index=False)
    return df_append
    
    
    