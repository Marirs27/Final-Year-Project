import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil
import cv2
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile                            

BASE_MODEL_PATH = os.path.join(os.getcwd(),"model")
PICKLE_DIR = os.path.join(os.getcwd(),"pickle_files")
JSON_DIR = os.path.join(os.getcwd(),"json_files")

if not os.path.exists(JSON_DIR):
    os.makedirs(JSON_DIR)

BEST_MODEL = "model/self_trained/distracted-25-0.99.hdf5"
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(64, 64))
    # convert PIL.Image.Image type to 3D tensor with shape (64, 64, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 64, 64, 3) and return 4D tensor
    return np.vstack([np.expand_dims(x, axis=0)])

ImageFile.LOAD_TRUNCATED_IMAGES = True 

def predict_result(image_tensor):

    ypred_test = model.predict(image_tensor,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    print(ypred_class)

    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
    ypred_class = int(ypred_class)
    print(id_labels[ypred_class])


    #to create a human readable and understandable class_name 
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


    with open(os.path.join(JSON_DIR,'class_name_map.json'),'w') as secret_input:
        json.dump(class_name,secret_input,indent=4,sort_keys=True)

    with open(os.path.join(JSON_DIR,'class_name_map.json')) as secret_input:
        info = json.load(secret_input)
        label = info[id_labels[ypred_class]]
        print(label)
    
    return label

INPUT_VIDEO_FILE = "input_video.mp4"
OUTPUT_VIDEO_FILE = "final_output_video-2.avi"

vs = cv2.VideoCapture(INPUT_VIDEO_FILE)
writer = None
(W, H) = (None, None)
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (128, 128))
	frame = np.expand_dims(frame,axis=0).astype('float32')/255 - 0.5
#	frame = cv2.resize(frame, (64,64))
#	frame = np.expand_dims(image.img_to_array(frame),axis=0).astype('float32')/255 - 0.5
#	model_base = VGG16(include_top=False)
#	test_vgg16 = model_base.predict(frame,verbose=1)
#   label = predict_result(test_vgg16)
	label = predict_result(frame)
    

	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
	
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE,fourcc, 30, (W,H), True)

	writer.write(output)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

print("[INFO] cleaning up...")
writer.release()
vs.release()