import cv2
import os
import numpy as np
import math
import face_recognition

# def faceDetection(test_img):
#     gray_img= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)

#     face_haar_cascade = cv2.CascadeClassifier("./HaarCascade/haarcascade_frontalface_default.xml")
#     faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3, minNeighbors=7)

#     return faces,gray_img

# x=faceDetection(cv2.imread("C:/Users/prath/Desktop/pythonProject3/trainingImages/0/A4.jpeg"))

def label_for_training_data(directory):
    faces=[]
    facesID=[]

    for path,subdirnames,filenames in os.walk(directory):
        # print(path,"path")
        # print(subdirnames,"subdirnames")
        # print(filenames,"filenames")
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("img_path", img_path)
            print("id", id)
            test_img = cv2.imread(img_path)
            print(test_img)
            if test_img is None:
                print("Img is not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(test_img)
            print(len(faces_rect),"faces_rect")
            if len(faces_rect)!=1:
                continue
            x,y,w,h = faces_rect[0]
            roi_gray = gray_img[y:y+w, x:x+h]
            faces.append(roi_gray)
            facesID.append(int(id))
    return faces, facesID

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img, face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0), thickness=5)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),3)
    
def create_dataset(faceId,name):
    vid = cv2.VideoCapture(0)
    currentFrame = 0
    parent_dir = "C:/Users/dell/OneDrive/Desktop/pythonProject3/trainingImages/"
    path = os.path.join(parent_dir,faceId)
    if not os.path.exists(path):
        os.makedirs(path)

    while (True):
        success,frame = vid.read()
        cv2.imshow("output",frame)
        cv2.imwrite("./trainingImages/" + faceId + "/" + name  +str(currentFrame) + ".jpg",frame)
        currentFrame+=1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # webCam.release()
    cv2.destroyAllWindows()
# create_dataset("4","pratham")


def video_to_images(video_path, frames_per_second=1):
    cam = cv2.VideoCapture(video_path)
    frame_list = []
    frame_rate = cam.get(cv2.CAP_PROP_FPS) #video frame rate

    faceId = "11"
    name = "test"
    parent_dir = f'C:/Users/dell/OneDrive/Desktop/pythonProject3/trainingImages/'
    path = os.path.join(parent_dir,faceId) 
    if not os.path.exists(path):
        os.makedirs(path)

    # frame
    current_frame = 0
    currentFrame = 0
    

    if frames_per_second > frame_rate or frames_per_second == -1:
        frames_per_second = frame_rate
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:

            # if video is still left continue creating images
            file_name = f"./trainingImages/" + faceId + "/" + name  +str(currentFrame) + ".jpg"
            print ('Creating...' + file_name)
            # print('frame rate', frame_rate)



            if current_frame % (math.floor(frame_rate/frames_per_second)) == 0:

                # adding frame to list
                frame_list.append(frame)

                # writing selected frames to images_path
                cv2.imwrite(file_name, frame)
                currentFrame+=1
    
    
            # increasing counter so that it will
            # show how many frames are created
            current_frame += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return frame_list

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# k=video_to_images("D:/opencvVideo/vid1.mp4",1)
# print(k)
# x,y= faceDetection(cv2.imread("D:/ABD/abd1.png"))
# print(x,y)