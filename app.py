import streamlit as st

from recorder import *
from recognizer import *


if st.radio("run live"):
    obj = Detect_verify()
    obj.capture_live()
    obj.process_frames()

    obj = Recognize_verify()
    passenger_list = obj.verify_faces()


    st.text(passenger_list)
        

