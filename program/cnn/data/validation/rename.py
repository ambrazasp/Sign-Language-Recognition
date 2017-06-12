#!/usr/bin/env python2
import numpy as np
import cv2
from scipy.signal import savgol_filter
import random
import os


if __name__ == "__main__":
    i = 0
    fileDir = 'B'
    for fileName in os.listdir(fileDir):
        #if fileName[0] != fileDir:
        _, extension = fileName.split('.')  
        i += 1 
        os.rename('{}/{}'.format(fileDir, fileName), '{}/{}.{}.{}'.format(fileDir, fileDir, i, extension))
        print("{}: {}".format(fileName, i))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
