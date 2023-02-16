# import the opencv library
import cv2
from ultralytics import YOLO
import os
import sys
import pandas as pd
import random
import numpy as np
# import pytesseract
# from pytesseract import image_to_string
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import keras_ocr
# import easyocr
# working_img_ind_list = [17, 144, 260, 120, 9, 23, 50, 313, 327, 346, 68]
# define a video capture object
# vid = cv2.VideoCapture(0)
# img_path = 'C:/prajesh/ML/billboard_data/archive/mark3/RoadEye-2/test/images/'
img_path = 'C:/prajesh/ML/billboard_data/archive/Open-Image-6/test/images/'

model = YOLO("C:/prajesh/ML/billboard_data/archive/mark3/RoadEye-2/best.pt")
pipline = keras_ocr.pipeline.Pipeline() #Creting a pipline 
# text_reader = easyocr.Reader(['en'])
img_list = []
for f in os.listdir(img_path):
    # print(img_path+f)
    # sys.exit()
    img_list.append(cv2.imread(img_path+f))
img_list_len = len(img_list)
tested_ind = []
working_img_ind = []
while True:
    check_ind = random.randint(0,img_list_len-1)
    if check_ind not in tested_ind:
        tested_ind.append(check_ind)
        test_img = img_list[check_ind]
        # test_img = cv2.imread('C:/prajesh/ML/billboard_data/archive/Billboards-3/test/images/m-plakat-siegestor-neu_png.rf.00822c3b1d0eb0886ce9d069f4c56f5c.jpg')
        # print(test_img)
        # sys.exit()
        # test_img = cv2.resize(test_img, (640, 480))
        # cv2.imshow("test", test_img)
        # cv2.waitKey(0)
        det = model.predict(test_img, conf=0.7, show=False)
        # print(det[0].boxes.boxes)
        # cv2.waitKey(0)
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
                # cv2.imshow("final", test_img)
                # cv2.waitKey(0)

                cropped_test = test_img[int(b[1]):int(b[1])+width, int(b[0]):int(b[0]) + height]
                # cv2.imshow("cropped", cropped_test)
                # cv2.waitKey(0)
                cropping_list.append(cropped_test)
            for i in cropping_list:
                # img_text_data = image_to_string(i)
                # img_text_data = text_reader.readtext(i, detail = 0, paragraph=True)
                img_text_data = pipline.recognize([i])[0]
                temp_string = ' '
                for j in range(len(img_text_data)):
                    temp_string = temp_string + img_text_data[j][0] + ' '
                    # print(img_text_data[j][0])
                print(temp_string)
                test_img = cv2.resize(test_img, (1280, 720))
                cv2.imshow("cropped_list", test_img)
                cv2.waitKey(0)
                # read_text.clear()
    if len(tested_ind) == img_list_len:
        print(working_img_ind)
        break  
    
    
    # https://meet.google.com/psd-goke-qwb