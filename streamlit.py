import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil
import cv2
from keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile     
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt 
import time                       

BASE_MODEL_PATH = 'model'
PICKLE_DIR = 'pickle_files'

BEST_MODEL = os.path.join(BASE_MODEL_PATH,"self_trained","distracted-25-0.99.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(img_path, target_size=(128, 128))
    img = np.asarray(img_path)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    # convert PIL.Image.Image type to 3D tensor with shape (128, 128, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 128, 128, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def return_prediction(filename):
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    print(type(filename))
    test_tensors = path_to_tensor(filename).astype('float32')/255 - 0.5

    ypred_test = model.predict(test_tensors,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)

    print(ypred_class)
    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    print(id_labels)
    ypred_class = int(ypred_class)
    res = id_labels[ypred_class]

    class_name = dict()
    class_name["c0"] = "SAFE_DRIVING"
    class_name["c1"] = "TEXTING_RIGHT"
    class_name["c2"] = "TALKING_PHONE_RIGHT"
    class_name["c3"] = "TEXTING_LEFT"
    class_name["c4"] = "TALKING_PHONE_LEFT"
    class_name["c5"] = "OPERATING_RADIO"
    class_name["c6"] = "DRINKING"
    class_name["c7"] = "REACHING_BEHIND"
    class_name["c8"] = "HAIR_AND_MAKEUP"
    class_name["c9"] = "TALKING_TO_PASSENGER"

    prediction_result = class_name[res]
    return prediction_result

st.title("Distracted Driver Detection")

fig = plt.figure()

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = return_prediction(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                # st.pyplot(fig)


if __name__=='__main__':
    main()