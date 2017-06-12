#!/usr/bin/env python2
import numpy as np
import cv2
from scipy.signal import savgol_filter
import random
import os


if __name__ == "__main__":
    fileDir = 'data/validate/'
    for fileFolder in os.listdir(fileDir):
        for fileName in os.listdir('{}/{}'.format(fileDir,fileFolder)):
            if fileName[0] != fileDir: 
                os.remove('{}/{}/{}'.format(fileDir,fileFolder,fileName))
                print("{}".format(fileName))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
