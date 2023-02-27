import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def show_detections(segmentations, detections):
    detections = np.where(detections >= 0.32, 0.5, -1)
    segmentations[0:40] = 0
    segmentations[-40:] = 0
    x = np.arange(segmentations.shape[0])/120
    barx = np.ones(segmentations.shape[0])

    fig = plt.figure(figsize=(13,10))
    ax_1 = plt.subplot(311)

    ax_1.spines["top"].set_visible(False)
    ax_1.spines["bottom"].set_visible(False)
    ax_1.spines["right"].set_visible(False)
    ax_1.spines["left"].set_visible(False)
    ax_1.get_xaxis().tick_bottom()
    ax_1.get_yaxis().set_visible(False)
    ax_1.set_ylim(0, 1.4)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    ax_1.text(0, 1.2, "Goals", fontsize=20, fontstyle="italic")

    ax_1.barh(barx-0.5, x, 0.4)
    ax_1.plot(x, detections[:, 0], 'y*', markersize=20)

    ax_2 = plt.subplot(312)
    ax_2.spines["top"].set_visible(False)
    ax_2.spines["bottom"].set_visible(False)
    ax_2.spines["right"].set_visible(False)
    ax_2.spines["left"].set_visible(False)
    ax_2.get_xaxis().tick_bottom()
    ax_2.get_yaxis().set_visible(False)
    ax_2.set_ylim(0, 1.4)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    ax_2.set_ylabel("Highlights", fontsize=20, color="tab:orange")
    ax_2.text(0, 1.2, "Cards", fontsize=20, fontstyle="italic")

    ax_2.barh(barx - 0.5, x, 0.4)
    ax_2.plot(x, detections[:, 1], 'y*', markersize=20)

    ax_3 = plt.subplot(313)
    ax_3.spines["top"].set_visible(False)
    ax_3.spines["bottom"].set_visible(False)
    ax_3.spines["right"].set_visible(False)
    ax_3.spines["left"].set_visible(False)
    ax_3.get_xaxis().tick_bottom()
    ax_3.get_yaxis().set_visible(False)
    ax_3.set_ylim(0, 1.4)
    plt.xticks([0, 10, 20, 30, 40, 50], fontsize=20)
    ax_3.text(0, 1.2, "Substitutions", fontsize=20, fontstyle="italic")
    ax_3.barh(barx - 0.5, x, 0.4)
    ax_3.plot(x, detections[:, 2], 'y*', markersize=20)
    ax_3.set_xlabel("Game Time (in minutes)", fontsize=20)
    st.pyplot(fig)

    st.success(f"💿  Successfully generated timeline for video")

    return