import cv2
import dlib
import sys
import numpy as np

#cap = cv2.VideoCapture(0)
imagePath = sys.argv[1]
image_basic = cv2.imread(imagePath)

face_detector = dlib.get_frontal_face_detector()

dlib_facelanadmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
gray = cv2.cvtColor(image_basic, cv2.COLOR_BGR2GRAY)

# #Initialize FAST detector
# star = cv2.xfeatures2d.StarDetector_create()

# #Initialize BRIEF extractor
# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# #find keypoints
# kp = star.detect(image_basic, None)

# #compute descriptors
# kp, des = brief.compute(image_basic, kp)

# print(brief.descriptorSize())
# print(des.shape)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold = 50000)

while True:
    #_, frame = cap.read()
    gray = cv2.cvtColor(image_basic, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelanadmark(gray, face)
        right_eye = []
        left_eye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x,y))
            #next = n + 1
            cv2.circle(image_basic, (x,y), 1, (0,0,255), 2)


        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x,y))
            #next = n + 1
            cv2.circle(image_basic, (x,y), 1, (0,0,255), 2)

        
    cv2.imshow('Face', image_basic)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

