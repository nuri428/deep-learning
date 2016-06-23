# -*- coding: UTF-8 -*-

__author__ = 'nuri'
import rawpy
import imageio
import os, glob, sys
import cv2

rawextens = ['.nef', '.cr',]
jpgextens = ['.jpg', '.jpeg',]

def scandirs(path):
    for currentFile in glob.glob( os.path.join(path, '*') ):
        if os.path.isdir(currentFile):
            print 'got a directory: ' + currentFile
            scandirs(currentFile)

        filename, file_extension = os.path.splitext(currentFile)
        file_extension = file_extension.lower()
        if  file_extension in rawextens:
            print "processing file: " + currentFile
            imageFilename = "{0}.jpg".format(filename)
            raw = rawpy.imread(currentFile)
            rgb = raw.postprocess()
            imageio.imsave(imageFilename, rgb)

        # if file_extension in jpgextens:
        #     print "processing file: " + currentFile
        #     imageFilename = "{0}.png".format(filename)
        #     image = cv2.imread(currentFile)
        #     cv2.imwrite(imageFilename, image )

#cvtRoot = "J:\\사진\\P&I\\2012\\삼성"
#scandirs(cvtRoot)
#sys.argv[1]

scandirs(sys.argv[1])
