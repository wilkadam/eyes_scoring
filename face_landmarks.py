

import cv2
import dlib
import sys
import numpy as np
from scipy.spatial import distance
import math
import os

# assign directory
directory = r"C:\Users\Adam\Documents\EyeImages"

listOfDifs = []
fileslist = []
# itrate over files in
# that directory
for filename in os.listdir(directory):
   # imagePath2 = str(os.path.join(directory, filename))
    #fileslist.append(imagePath2)

    #print(imagePath2)

    #print(fileslist)
    imagePath = sys.argv[1]
    image_basic = cv2.imread(imagePath)
    scale_percent = 100

    width = int(image_basic.shape[1] * scale_percent / 100)
    height = int(image_basic.shape[0] * scale_percent / 100)

    dsize = (width, height)
    image_basic_resized = cv2.resize(image_basic, dsize)

    imagePath2 = sys.argv[2]
    image_to_compare = cv2.imread(imagePath2)
    image_to_compare_resized = cv2.resize(image_to_compare, dsize)

    face_detector = dlib.get_frontal_face_detector()
    face_detector1 = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dlib_facelandmark1 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image_basic_resized, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image_to_compare_resized, cv2.COLOR_BGR2GRAY)

    while True:

        faces = face_detector(gray)
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            right_eye = []
            left_eye = []
            distances = []
            
            a_values = [1, 1, 18, 39, 8, 58, 58, 31, 31]
            b_values = [49, 16, 26, 47, 28, 26, 18, 5, 13]

            for a,b in zip(a_values, b_values):
                x = face_landmarks.part(a).x
                y = face_landmarks.part(a).y
                x1 = face_landmarks.part(b).x
                y1 = face_landmarks.part(b).y
                start_point = (x,y)
                end_point = (x1,y1)
                cv2.line(image_basic_resized, start_point, end_point, (0,0,255), 2)
                distance = math.dist(start_point, end_point)
                distances.append(distance)

        faces2 = face_detector1(gray1)
        for face2 in faces2:
            face_landmarks = dlib_facelandmark1(gray1, face2)
            right_eye = []
            left_eye = []
            distances1 = []
            
            c_values = [1, 1, 18, 39, 8, 58, 58, 31, 31]
            d_values = [49, 16, 26, 47, 28, 26, 18, 5, 13]

            for a,b in zip(c_values, d_values):
                z = face_landmarks.part(a).x
                v = face_landmarks.part(a).y
                z1 = face_landmarks.part(b).x
                v1 = face_landmarks.part(b).y
                start_point1 = (z,v)
                end_point1 = (z1,v1)
                cv2.line(image_to_compare_resized, start_point1, end_point1, (0,0,255), 2)
                distance1 = math.dist(start_point1, end_point1)
                distances1.append(distance1)

        cv2.imshow('Face', image_basic_resized)
        cv2.imshow('Face2', image_to_compare_resized)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

    print(distances)
    print(distances1)

    total_diff = 0
    total_sum = 0
    total_lenght_base = 0
    total_length_second = 0
    ints = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    values_diffrences = []

    for num1, num2, wart in zip(distances, distances1, ints):
        abs_difference = (abs(num1-num2))
        total_diff += abs_difference 
        total_lenght_base += num1
        total_length_second += num2
        total_sum += num1
        diff = 0.1 * total_sum
        difference_of_certain_value_different = 1 - (abs_difference / num1)
        values_diffrences.append(difference_of_certain_value_different)
        print("Wartość numer " + str(wart) + " wynosi: " + str(difference_of_certain_value_different))
        # if (total_diff < diff):
        #     print("Fitted in 90%!")
        # elif (abs_difference < lol_value1):
        #     print("Fitted in 80%!")
        # else:
        #     print("Not fitted")

    #Difference in summary length
    difference = abs(total_lenght_base - total_length_second)
    percent = 1 - (difference / total_lenght_base)
    biggest_difference = values_diffrences.index(min(values_diffrences)) + 1
    smallest_difference = values_diffrences.index(max(values_diffrences)) + 1

    print("Najbardziej różniąca się wartość: " + str(biggest_difference))
    print("Najmniej różniąca się wartość: " + str(smallest_difference))
    #print("Najbardziej rózniąca się długość: " + str(difference_of_certain_value_different.index(min(difference_of_certain_value_different))))
    print("Całkowita róznica długości: " + str(difference))
    print("Procent całkowiego podobienstwa: " + str(percent) + "%")


    cv2.destroyAllWindows()


#Porownywac na zasadzie, czy jest podobny w ilu procentach