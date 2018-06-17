import cv2
import numpy as np
import os
#for jpeg type image file ext
jpgextens = ['.jpg', '.jpeg']
imgextens = ['.jpg','.jpeg', '.png','.bmp']
resizes = [1600, 1280, 1024, 640, 480]

def rotateImage(image, angle):
    row = image.shape[0]
    col = image.shape[1]
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def get_fileName(filePath):
    filePath = os.path.basename(filePath)
    filePath=filePath[:filePath.index(".")]
    return filePath