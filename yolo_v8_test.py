# import the opencv library
import cv2
from ultralytics import YOLO
import os
import sys
import pandas as pd
import random
import numpy as np
import keras_ocr
img_path = 'C:/prajesh/ML/billboard_data/archive/Open-Image-6/test/images/'

model = YOLO("C:/prajesh/ML/billboard_data/archive/mark3/RoadEye-2/best.pt")
pipline = keras_ocr.pipeline.Pipeline() #Creting a pipline 
img_list = []
for f in os.listdir(img_path):
    img_list.append(cv2.imread(img_path+f))
img_list_len = len(img_list)
tested_ind = []
working_img_ind = []
while True:
    check_ind = random.randint(0,img_list_len-1)
    if check_ind not in tested_ind:
        tested_ind.append(check_ind)
        test_img = img_list[check_ind]
        det = model.predict(test_img, conf=0.7, show=False)
        cropping_list = []
        bbox = np.asarray(det[0].boxes.boxes)
        if len(bbox) > 0:
            working_img_ind.append(check_ind)
            for b in bbox:
                start_point = (int(b[0]), int(b[1]))
                end_point =  (int(b[2]), int(b[3]))
                print("start_point->", start_point)
                print("end_point->", end_point)
                width = int(b[3]) - int(b[1])
                height = int(b[2]) - int(b[0])
                test_img = cv2.rectangle(test_img, start_point, end_point, color=(0, 255, 0), thickness=2)
                cropped_test = test_img[int(b[1]):int(b[1])+width, int(b[0]):int(b[0]) + height]
                cropping_list.append(cropped_test)
            for i in cropping_list:
                img_text_data = pipline.recognize([i])[0]
                temp_string = ' '
                for j in range(len(img_text_data)):
                    temp_string = temp_string + img_text_data[j][0] + ' '
                print(temp_string)
                test_img = cv2.resize(test_img, (1280, 720))
                cv2.imshow("cropped_list", test_img)
                cv2.waitKey(0)
    if len(tested_ind) == img_list_len:
        print(working_img_ind)
        break  