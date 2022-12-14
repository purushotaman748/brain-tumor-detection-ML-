from keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
import streamlit as st

file = "model.h5"

loaded_model = load_model(file)


def names(number):
    if number == 0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


def predict(IMG):
    img = Image.open(IMG)
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = loaded_model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    st.write(str(res[0][classification] * 100) + "% confidence that " + names(classification))
