#!/usr/bin/env python2
import numpy as np
import cv2
from scipy.signal import savgol_filter
import random
import os


if __name__ == "__main__":
    i = 360
    fileDir = 'X'
    for fileName in os.listdir(fileDir):
        if fileName[0] != fileDir:        
            i += 1 
            os.rename('{}/{}'.format(fileDir, fileName), '{}/{}.{}.png'.format(fileDir, fileDir, i))
            print("{}: {}".format(fileName, i))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
