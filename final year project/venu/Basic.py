import cv2
import numpy as np
import face_recognition

imgScrl = face_recognition.load_image_file('Known/Scarlett Johansson.jpeg')
imgScrl = cv2.cvtColor(imgScrl,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Unknown/Natasha.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgScrl)[0]
encodeScrl = face_recognition.face_encodings(imgScrl)[0]
cv2.rectangle(imgScrl,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(50,50,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(50,50,255),2)
 
results = face_recognition.compare_faces([encodeScrl],encodeTest)
faceDis = face_recognition.face_distance([encodeScrl],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,0,255),2)
 
cv2.imshow('Scarlett Johansson',imgScrl)
cv2.imshow('Scarlett Johansson Test',imgTest)
cv2.waitKey(0)