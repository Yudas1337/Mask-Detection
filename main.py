import hashlib
import os
import threading

import cv2
import numpy as np
import pyttsx3
from gtts import gTTS
from keras.models import load_model
from playsound import playsound
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# text to speech
class Speech:
    def __init__(self):
        self.die = False
        self.busy = False
        self.tts = pyttsx3.init()
        self.tts.setProperty('voice', 'indonesian')

    def say(self, txt, using_gtts=True):
        if self.busy:
            return

        self.busy = True

        if using_gtts:
            sfname = ''.join(x for x in txt if x.isalnum())
            sfname = sfname[0:3] + sfname[-3:] + hashlib.md5(sfname.encode()).hexdigest()
            sfname = 'speech/{}.mp3'.format(sfname)

            if not os.path.exists(sfname):
                try:
                    speech = gTTS(txt, lang='id')
                    speech.save(sfname)
                except:
                    os.unlink(sfname)
                else:
                    playsound(sfname)
            else:
                playsound(sfname)

        self.busy = False



def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))


    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()


    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []


    # loop detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]


        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5: #"confidence":
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)


            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))


    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# ------------------------------------------
if __name__ == '__main__':


    labels_dict={0:'without mask', 1:'mask'}
    color_dict={0:(0,255,0), 1:(0,0,255)}
    displayTxt = ''


    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(['models/face_detector', 'deploy.prototxt'])
    weightsPath = os.path.sep.join(['models/face_detector',
        'res10_300x300_ssd_iter_140000.caffemodel'])


    # load model
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model('models/mask_detector/mask_detector.model')


    # instance from Speech class
    onmask = 0
    speech = Speech()


    # print('starting video captures..')
    webcam = cv2.VideoCapture(0) #Use camera 0


    # resize image to speed analyse
    resize = 1/4


    # loop over the frames from the video stream
    while True:


        # grab the frame from the threaded video stream and resize it
        (rval, im) = webcam.read()
    
        #resize and flip mirror
        im = cv2.flip(im, 1)
        
        # detect and predict mask
        (locs, preds) = detect_and_predict_mask(im, faceNet, maskNet)


        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if mask > withoutMask:
                displayTxt = 'Masked'
                label = 0 #'mask'
                onmask += 1 
                if onmask >= 10:
                    onmask = 0
                    txt = 'Silahkan Masuk'
                    sp = threading.Thread(target=speech.say, args=(txt,))
                    sp.start()
                    
            else:
                displayTxt = 'No Mask'
                label = 1
                onmask -= 1
                if onmask <= -10:
                    onmask = 0
                    txt = 'Harap gunakan masker sekarang'
                    sp = threading.Thread(target=speech.say, args=(txt,))
                    sp.start()
            
            displayTxt = "{}: {:.2f}%".format(displayTxt, max(mask, withoutMask) * 100)
            
            cv2.putText(im, displayTxt, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dict[label], 2)
            cv2.rectangle(im,(startX,startY),(endX,endY),color_dict[label],2)


        # show the output frame
        cv2.imshow("TI-3C Kelompok 7 Annisa_Yudas", im)


        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    try:
        sp.join()
    except:
        pass


    webcam.release()
    cv2.destroyAllWindows()