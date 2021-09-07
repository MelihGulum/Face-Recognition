import numpy as np
import cv2
import face_recognition
import os


path ='known'
images= []
class_name=[]
my_list= os.listdir(path)
#print(my_list)

for cl in my_list:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    class_name.append(os.path.splitext(cl)[0])
print(class_name)


def find_encodings(images):
    encode_list =[]
    for img in images:
        encode = face_recognition.face_encodings(img,None,2)[0]
        encode_list.append(encode)
    return encode_list


encode_list_known = find_encodings(images)
#print(len(encode_list_known))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encode_list_known,encodeFace,0.6)
        faceDis = face_recognition.face_distance(encode_list_known,encodeFace)
        #print(faceDis)
        #print(matches)
        matchIndex = np.argmin(faceDis)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        if matches[matchIndex]:
            name = class_name[matchIndex].upper()
            #print(name)
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,f'{name}',(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)


    cv2.imshow('webcam', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # press  q and quit if you dont it can not close
        break

