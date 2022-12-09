import streamlit as st
from io import BytesIO
from model import predict

st.title("Brain tumor detection software")

STYLE = """
<style>
img {
    max-width:100%
}
</style>
"""

st.markdown("This software automatically detects **tumor from an  MRI scan**")
st.markdown(STYLE, unsafe_allow_html=True)

file = st.file_uploader("Upload MRI", type=["png", "jpeg", "jpg"])
show_file = st.empty()
try:
    if isinstance(file, BytesIO):
        show_file.image(file)
        predict(file)

except:
    st.write("""
    please resize image...
    Image not compatible with software""")
