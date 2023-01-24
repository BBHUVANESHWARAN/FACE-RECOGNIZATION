
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from pymongo import MongoClient
from tkinter import messagebox
import pandas as pd
import tkinter as tk
import smtplib
import random

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

dataframe=pd.read_csv('details.csv')

card_number=int(input('Enter Your Number:--'))

name_list=[]

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        cv2.destroyAllWindows()
        for index,value in dataframe.iterrows():
            if value['Name']==name and value['card_number']==card_number:
                root=tk.Tk()
                root.geometry("600x400")

                name_var=tk.StringVar()
                passw_var=tk.StringVar()

                def Close():
                    root.destroy()

                def openNewWindow():
                    name=int(name_var.get())
                    if name>1000:
                        newWindow=tk.Toplevel(root)
                        newWindow.title("New Window")
                        newWindow.geometry("400x250")
                        sub_btn=tk.Button(newWindow,text = 'Exit',command=Close)
                        sub_btn.pack(pady=10)

                    else:
                        messagebox.showwarning("showwarning", "in sufficient balance")
                        root.destroy()

                name_label = tk.Label(root, text = 'Enter Your Amount', font=('calibre',10, 'bold'),anchor='nw')
                name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
                sub_btn=tk.Button(root,text = 'ok', command = openNewWindow)
                name_label.grid(row=0,column=0)
                name_entry.grid(row=0,column=1)
                sub_btn.grid(row=3,column=1)
                root.mainloop()
                break
                
            else:
                message=str(random.randint(999,9999))
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login("joviyalarun07@gmail.com", "")
                s.sendmail("joviyalarun07@gmail.com", "ananthakumar9794@gmail.com", message)
                s.quit()
                
                OTP=input('enter your OTP:-')
                
                if OTP==message:
                    root=tk.Tk()
                    root.geometry("600x400")

                    name_var=tk.StringVar()
                    passw_var=tk.StringVar()

                    def Close():
                        root.destroy()

                    def openNewWindow():
                        name=int(name_var.get())
                        if name>1000:
                            newWindow=tk.Toplevel(root)
                            newWindow.title("New Window")
                            newWindow.geometry("400x250")
                            sub_btn=tk.Button(newWindow,text = 'Exit',command=Close)
                            sub_btn.pack(pady=10)

                        else:
                            messagebox.showwarning("showwarning", "in sufficient balance")
                            root.destroy()

                    name_label = tk.Label(root, text = 'Enter Your Amount', font=('calibre',10, 'bold'),anchor='nw')
                    name_entry = tk.Entry(root,textvariable = name_var, font=('calibre',10,'normal'))
                    sub_btn=tk.Button(root,text = 'ok', command = openNewWindow)
                    name_label.grid(row=0,column=0)
                    name_entry.grid(row=0,column=1)
                    sub_btn.grid(row=3,column=1)
                    root.mainloop()
                    break
                    
                    
                

        break

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
