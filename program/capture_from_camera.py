#!/usr/bin/env python2
"""
Contains code to capture images from a live webcam recording.
"""

import cv2
import os
import time
def save_image(frame, path):
    #cv2.imshow("win", frame)
    print("Writing the captured image to file...")
    ret = cv2.imwrite(path, frame)
    if not ret:
        print("Error in writing the image to the file path '{}'".format(
            path))
    print("Done!")


def main():
    camera = cv2.VideoCapture(0)
    num_images = 15
    num_seconds_to_wait = 1
    time.sleep(num_seconds_to_wait)
    folder = 'Y'
    output_file_path = 'captures/{}'.format(folder)
    #for fileName in os.listdir(folder):
        #output_file_path = os.path.join(folder, fileName)
    for num_image in xrange(1, num_images + 1):
        print(folder)
        time.sleep(num_seconds_to_wait)
        print("\n\nTaking image #{}...".format(num_image))
        output_file_name = "{}/{}{}.png".format(output_file_path, folder, num_image)            
        #capture_images(camera, output_file_name)
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image!")
        print("Writing the captured image to file...")
        ret = cv2.imwrite(output_file_name, frame)
        if not ret:
            print("Error in writing the image to the file path '{}'".format(
                output_file_name))
        print("Done!")
    print "\n\nReleasing the camera..."
    camera.release()
    cv2.destroyAllWindows()
    print "The program completed successfully !!"


if __name__ == '__main__':
    main()
