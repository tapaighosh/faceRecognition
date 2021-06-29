import cv2
import numpy as np
import face_recognition
import os
import csv
import datetime

path = 'Known'
images = []
ClassNames = []
myList = os.listdir(path)  # all the image in the list
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # readin current image
    images.append(curImg)  # append image in images
    # append name ex - Chrish Hemsworth
    ClassNames.append(os.path.splitext(cl)[0])
# print(ClassNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert in to Rgb image
        encode = face_recognition.face_encodings(img)[0]  # encodeing the faces
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complet')


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        print(nameList)
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


cap = cv2.VideoCapture(0)  # capturing video

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # redusing the size 1/4
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(
        imgS)  # find the current face location
    encodesCurFrame = face_recognition.face_encodings(
        imgS, facesCurFrame)  # encode the current face

    # encodeface and currloc want in the same time useing zip
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matchs = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matchs[matchIndex]:
            name = ClassNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
