import sys
import os
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from audiorecorder import audiorecorder
# Add main directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from progression_score.create_score import create_progression_score
    
def main():
    st.title("Parkinson's Progression Tracking")

    # Medication status input
    st.header("Medication Status")
    medication_status = st.selectbox("Are you currently on medication?", ["On medication", "Off medication"])

    if medication_status.lower() not in ["on medication", "off medication"]:
        st.warning("Please select either 'On medication' or 'Off medication' for medication status.")
        return

    # DBS status input
    st.header("DBS Status")
    # If participant with PD: either "On DBS" (deep brain stimulator switched on or within 1 hour of it being switched off), "Off DBS" (1 hour or longer after deep brain stimulator switched off until it is switched back on again) or "-" (no deep brain stimulator in situ).
    st.write("Deep Brain Stimulation (DBS) is a surgical procedure used to treat several disabling neurological symptoms—most commonly the debilitating motor symptoms of Parkinson’s disease (PD), such as tremor, rigidity, stiffness, slowed movement, and walking problems.")
    st.write(
            "If you have been diagnosed with PD and have undergone DBS surgery, select:\n"
            "On DBS: Deep brain stimulator switched on or within 1 hour of it being switched off,\n"
            "Off DBS: 1 hour or longer after deep brain stimulator switched off until it is switched back on again, or\n"
            "-: Diagnosed with PD, no DBS surgery.\n"
            "If you have not been diagnosed with PD, please select Control."
    )
    dbs_status = st.selectbox("Do you have Deep Brain Stimulation (DBS)?", ["Control", "On DBS", "Off DBS", "-"])

    if dbs_status.lower() not in ["control", "on dbs", "off dbs", "-"]:
        st.warning("Please enter either 'Control', 'On DBS', 'Off DBS', or '-' for DBS status.")
        return

    # Drawing input
    st.header("Draw a Picture")
    canvas_result = st_canvas(
        stroke_width=5,
        stroke_color="#000000",  # Black drawing color
        background_color="#ffffff",  # White background color
        height=256,  # Set canvas height to 256
        width=256,  # Set canvas width to 256
        drawing_mode="freedraw",
        key="canvas",
    )
    drawing = canvas_result.image_data

    # Convert the drawing to PNG format
    if drawing is not None:
        image = Image.fromarray(drawing.astype('uint8'), 'RGBA')
        image.save("drawing.png")

    # Voice input
    st.header("Audio Recorder")
    audio = audiorecorder("Click to record", "Click to stop recording")

    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.export().read())  

        # To save audio to a file, use pydub export method:
        audio.export("audio.wav", format="wav")

        # To get audio properties, use pydub AudioSegment properties:
        st.write(f"Frame rate: {audio.frame_rate}, Duration: {audio.duration_seconds} seconds")

    # Video input
    st.header("Upload a Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    if video_file is not None:
        # Display the uploaded video
        st.video(video_file)

        # Save the uploaded video as an MP4 file
        save_path = "uploaded_video.mp4"
        with open(save_path, "wb") as f:
            f.write(video_file.read())

    if st.button("Classify"):
        if drawing is None or audio is None or video_file is None:
            st.warning("Please provide all inputs")
        else:
            prediction = create_progression_score('drawing.png', 'audio.wav', 'uploaded_video.mp4', medication_status, dbs_status)
            st.success(f"Parkinson's Progression Score: {round(prediction*100, 2)}%")

if __name__ == "__main__":
    main()
