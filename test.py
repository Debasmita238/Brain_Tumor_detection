import cv2
import os
from PIL import Image
import numpy as np

from keras.models import load_model

model= load_model('BrainTumor10E.h5')

image = cv2.imread('/home/debasmitasaha238/Brain_Tumor_Detection/pred/pred2.jpg')

img = Image.fromarray(image)

img= img.resize((64,64))
img = np.array(img)

input_img=np.expand_dims(img, axis=0)

result = model.predict(input_img)
print(result)



