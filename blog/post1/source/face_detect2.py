import imageio
import os, glob, sys
import cv2
import exifread
import numpy as np
import face_util as fu
# Get user supplied values


def getfaces2(imgPath, faceCascade, eyeCascade, targetPath, idx):
    newHeight = 256
    newWidth = 256
    filename = fu.get_fileName(targetPath)

    f = open(imgPath, 'rb')
    tags = exifread.process_file(f)
    f.close()

    angle = 0
    try :
        rotateString = "{0}".format(tags["Image Orientation"])
    except:
        rotateString = ""

    rSegStr = rotateString.split()
    #print rSegStr
    try :
        if len(rSegStr) > 1 and rSegStr[0].lower() == "rotated" :
            angle = float(rSegStr[1])
            if len(rSegStr)> 2 :
                if rSegStr[2] == "CW" :
                    angle *= -1
    except :
        print "exception happen string is {0}".format(rotateString)
        angle = 0

    # Read the image
    resizeIdx = idx;
    for resize_len in fu.resizes:
        image = cv2.imread(imgPath)
        height = image.shape[0]
        width = image.shape[1]

        if angle <> 0:
            image = fu.rotateImage(image, angle)
            height = image.shape[0]
            width = image.shape[1]

        if height > resize_len and width > resize_len :
            if height > width :
                r = resize_len * 1.0 / image.shape[0]
                dim = (resize_len, int(image.shape[1]*r))
            else :
                r = resize_len* 1.0 / image.shape[1]
                dim = (resize_len, int(image.shape[0]*r))
            try :
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            except :
                print "resize:{0}".format(resize_len)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        print "Found {0} faces!".format(len(faces))

        lidx = 0
        for (x, y, w, h) in faces:
            faceImg = image[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(roi_gray)
            if len(eyes) < 2 :
                continue
            lidx = lidx +1
            faceImg = cv2.resize(faceImg, (newHeight,newWidth))
            tPath = "{0}\\{1}_{2}_{3}_{4}.jpg".format(os.path.dirname(targetPath), filename, resizeIdx, lidx, len(eyes))
            cv2.imwrite(tPath, faceImg)
            print "file : {0}, eye Count : {1}".format(tPath, len(eyes))
        resizeIdx += 1


