# -*- coding: utf-8 -*-

import cv2
from pytesseract import pytesseract #outil de reconnaissance de caractères (OCR)
from pytesseract import Output
from tkinter import *
import tkinter as tk # création d'interfaces graphiques.
from tkinter import filedialog
import pandas as pd # l'analyse des données
import sys 
import os
import csv



# detection des objects (image)
def detection_objet_image():
    try :
        filepath=filedialog.askopenfilename(filetypes=[
                        ("image", ".jpeg"),
                        ("image", ".png"),
                        ("image", ".jpg"),
                    ])
        
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'

        model = cv2.dnn_DetectionModel (frozen_model,config_file)

        classLabels=[]
        file_name='Labels.txt'
        with open(file_name,'rt') as fpt:
            classLabels = fpt.read().rstrip('\n').split('\n')

        model.setInputSize(320,320)
        model.setInputScale(1.0/127.5)
        model.setInputMean((127.5,127.5,127.5))
        model.setInputSwapRB(True)

        img=cv2.imread(filepath)



        ClassIndex, confidece, bbox = model.detect(img,confThreshold=0.5)

        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN
        for ClassInd , conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(), bbox):
            cv2.rectangle(img,boxes,(255,0,0),2)
            cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale,color=(0,255,0),thickness=2)

        cv2.imshow("window",img)
        cv2.waitKey(0)
        
    except Exception:
        pass
        print("vous devez ajouter votre fichier !\n")


# detection des objects (video)
def detection_objet_video():
    try:
        filepath=filedialog.askopenfilename(filetypes=[
                        ("all video format", ".mp4"),
                        ("all video format", ".flv"),
                        ("all video format", ".avi"),
                    ])
        
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'

        model = cv2.dnn_DetectionModel (frozen_model,config_file)

        classLabels=[]
        file_name='Labels.txt'
        with open(file_name,'rt') as fpt:
            classLabels = fpt.read().rstrip('\n').split('\n')

    
        
        model.setInputSize(320,320)
        model.setInputScale(1.0/127.5)
        model.setInputMean((127.5,127.5,127.5))
        model.setInputSwapRB(True)
        
        cap=cv2.VideoCapture(filepath)

    
        #vérifiez si le videoCapture est ouvert correctement
        if not cap.isOpened():
            cap= cv2.videoCapture(0)
            
        if not cap.isOpened():
            raise IOError("impossible d'ouvrir la vidéo")
        
            
        font_scale=3
        font = cv2.FONT_HERSHEY_PLAIN

        
        while cap.isOpened():
            ret,frame = cap.read()
            
            ClassIndex, confidece, bbox = model.detect(frame,confThreshold=0.55)
            
            if (len(ClassIndex)!=0):
                for ClassInd ,conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(), bbox):
                    if(ClassInd<=80):
                        cv2.rectangle(frame,boxes,(255,0,0),2)
                        cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale,color=(0,255,0),thickness=2)


            cv2.imshow("détection d'objet",frame)
            
            if cv2.waitKey(2) & 0xFF == 27 :
                break
                
        
        
        cap.release()

    except Exception as e:
        print("la reconnaissance des objets dans la vidéo a été effectuée avec succès\n")
    


# detection de texte (image)
def detection_texte():
    try: 
        #chemin pour Windows
        #pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

        #chemin pour Linux
        pytesseract.tesseract_cmd = "/usr/bin/tesseract"

        filepath=filedialog.askopenfilename(filetypes=[
                        ("image", ".jpeg"),
                        ("image", ".png"),
                        ("image", ".jpg"),
                    ])
   
        

        img = cv2.imread(filepath)

        image_data = pytesseract.image_to_data(img, output_type=Output.DICT)
    

        for i, word in enumerate(image_data['text']):
            if word !="":
                x,y,w,h = image_data['left'][i],image_data['top'][i],image_data['width'][i],image_data['height'][i]
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.putText(img, word,(x,y-16),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0),1)

        
        cv2.imshow("window",img)
        cv2.waitKey(0)
        
        # ajouter le contenu de l'image dans un fichier .txt
        parse_text = []
        word_list = []

        last_word = ''

        for word in image_data['text']:

            if word!='':

                word_list.append(word)
                last_word = word

            if (last_word!='' and word == '') or (word==image_data['text'][-1]):

                parse_text.append(word_list)
                word_list = []

        with open('result_text.txt',  'w', newline="") as file:
            csv.writer(file, delimiter=" ").writerows(parse_text)

    except Exception:
        pass
        print("vous devez ajouter votre fichier !\n")

    

# detection de forme (image)

def detection_forme():

    try:
        filepath=filedialog.askopenfilename(filetypes=[
                        ("image", ".jpeg"),
                        ("image", ".png"),
                        ("image", ".jpg"),
                    ])
        
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0

        for contour in contours:

            if i == 0:
                i = 1
                continue

            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            # mettre le nom de la forme au centre de chaque forme
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif len(approx) == 4:
                cv2.putText(img, 'Quadrilateral', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif len(approx) == 5:
                cv2.putText(img, 'Pentagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif len(approx) == 6:
                cv2.putText(img, 'Hexagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            elif len(approx) == 7:
                cv2.putText(img, 'heptagone', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            elif len(approx) == 8:
                cv2.putText(img, 'octogone', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                

            else:
                cv2.putText(img, 'circle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('shapes', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception:
        pass
        print("vous devez ajouter votre fichier !\n")

    
# detection de couleurs (image)
def detection_couleur():

    os.system("python3 couleur.py")
    
       
# creation d'une fenetre pour choisir le mode !

def window1():


    window = tk.Tk()
    window.title("Reconnaissance de texte APP !")
    label=Label(window,text="choisissez votre mode de détection :")
    label.pack()
    label.place(x=170,y=1)

    window.geometry('520x300')

    window.minsize(520,300)
    window.maxsize(520,300)
    
    T1=Label(window,text="Détection d'objets [image] ",font=("Arial Bold",9))
    T1.place(x=1,y=40)
    button = Button(window,text='ouvrir le fichier', command=detection_objet_image,bg='light grey')
    button.pack(side=tk.BOTTOM)
    button.place(x=240,y=38)

     
    T2=Label(window,text="Détection d'objets [vidéo] ",font=("Arial Bold",9))
    T2.place(x=1,y=80)
    button2= Button(window,text="ouvrir le fichier", command=detection_objet_video,bg='light grey')
    button2.pack(side=tk.BOTTOM)
    button2.place(x=240,y=78)
    

    T3=Label(window,text="Reconnaissance de texte [image] ",font=("Arial Bold",9))
    T3.place(x=1,y=120)
    button3 = Button(window,text='ouvrir le fichier', command=detection_texte,bg='light grey')
    button3.pack(side=tk.BOTTOM)
    button3.place(x=240,y=116)

    

    T4=Label(window,text="Reconnaissance de formes [image] ",font=("Arial Bold",9))
    T4.place(x=1,y=160)
    button4= Button(window,text="ouvrir le fichier", command=detection_forme,bg='light grey')
    button4.pack(side=tk.BOTTOM)
    button4.place(x=240,y=154)

    
    T5=Label(window,text="Reconnaissance de couleurs [image] ",font=("Arial Bold",9))
    T5.place(x=1,y=200)
    button5 = Button(window,text='ouvrir le fichier', command=detection_couleur,bg='light grey')
    button5.pack(side=tk.BOTTOM)
    button5.place(x=240,y=192)


    
    button6= Button(window,text="Close", command=window.destroy, bg='red')
    button6.pack(side=tk.BOTTOM)
    button6.place(x=263,y=240)
     
    window.mainloop()

    

#MAIN : 


window1()