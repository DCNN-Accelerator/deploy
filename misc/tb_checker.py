import cv2 
import numpy as np 
import os

from utils.util import checkFPGAOutputs, generate_sobel,image_rescale,zeroPad

if __name__ == "__main__":

    path = os.getcwd() + r'\images\image_1.PNG'
    path_1 = os.getcwd() + r'\checkFile.txt'

    img = image_rescale(image_path=path, dim=512)
    img = zeroPad(img,7)
    cv2.imshow('thing',img)
    cv2.waitKey(0)

    sobel = generate_sobel(7)

    garbage = checkFPGAOutputs(input_image=img, kernel_float=sobel, outputStreamFileName=path_1,expectedDim=518)
    np.savetxt('garbage.txt',garbage)

