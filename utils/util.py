"""
Authors: Hussain Khajanchi and Paul Brodhead

Helper functions for Image Processing with the FPGA-based DCNN Accelerator

Includes modules for: 
    - Image resizing and color space conversion
    - Kernel Generation (sobel and identity, more to be added later)
    - Fixed Point quantization (using MATLAB Fixed-Point designer)
    - 1D Stream conversion and file generation
    - Output validation and display
"""

import cv2
import matlab.engine
from scipy.signal import correlate2d
import numpy as np 
import os


def image_rescale(image_path,dim):

    """
    Takes an input image path, and converts it to greyscale and the user dimensions (has to be square)
    @param image_path: string path containing the image file location
    """

    img = cv2.imread(image_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, [dim,dim])

def generate_sobel(kernel_size):

    """
    Creates a squar eSobel Edge-Detection filter to be used for image filtering on the FPGA
    
    @param kernel_size: odd filter dimension 

    """
    kernel = np.zeros([kernel_size,kernel_size])
    kernel[0:int(kernel_size/2),:] = -1 
    kernel[int(kernel_size)+1:kernel_size,:] = 1

    return kernel

def generate_identity(kernel_size):

    """
    Creates a square identity filter based on the kernel size parameter

    @param kernel_size: int specifying the kernel size
    Ex: 
    kernel_size = 3
      0 0 0 
      0 1 0
      0 0 0
    
    """

    kernel = np.zeros([kernel_size,kernel_size])
    kernel[int(kernel_size/2), int(kernel_size/2)] = 1
    return kernel

def quantize(image, kernel):

    """
    Uses MATLAB to quantize the image and kernel values into the specified fixed-point precision
    Requires a Licensed MATLAB installation with the Fixed-Point Designer add on 

    @param image: a NumPy array containing the image 
    @param kernel: a numpy array containing kernel data

    Returns the quantized kernel and image
    """

    eng = matlab.engine.start_matlab()
    return eng.fp_quantize(kernel, image)

def createUARTStream(image,kernel,streamFileName):

    """
    Creates a 1D stream containing kernel and image data and writes to a file
    Requires int8 quantized data

    @param image: numpy array containing image data
    @param kernel: numpy array containing kernel data
    @param streamFileName: string specifiying the filename that the stream should be written to 

    """

    stream = []

    # Flatten and append to list, starting with kernel
    stream.append(np.reshape(kernel,[kernel.shape[0]**2, 1]))
    stream.append(np.reshape(image, [image.shape[0]**2, 1]))
    
    x = np.array(stream,dtype=np.uint8)
    np.savetxt(streamFileName,x)

def zeroPad(image,kernel_size):

    """
    Takes the image, and zero pads it according to the kernel size
    @param image: nparray containing image data
    @param kernel_size: int containing the square size of the kernel

    """
    num_pad_zeros = int(kernel_size/2)
    return np.pad(image,((num_pad_zeros,num_pad_zeros),(num_pad_zeros,num_pad_zeros)),mode='constant')


def runFPGAConvolution(inputStreamFileName="uart_input_bytes.txt", cs_FileName="test_serial.cs", outputStreamFileName="fpgaOut.txt"): 

    """
    This function invokes the C# program to send/recieve convolution data from the FPGA 
    Given a stream path, this function compiles the C# program and invokes the executable
    
    This method will fail if the C# compiler is not part of the System Path

     
        @param: inputStreamFileName: a path to the file containing the 8-bit bytes to be streamed to the FPGA
        @param: outputStreamFileName: a path to the file containing the FPGA outputs
        @param: cs_FileName: a string designating the C# file to be compiled (should be testSerial.cs by default)

    For future: 
        - Replace File I/O with multiprocessing between Python and C# 

    Notes: 
        was having trouble getting C# cmd-line args to work so for now all the output files are "fpgaOut.txt", 
        this method ignores the outputStreamFileName param for now

    """
    os.system('powershell.exe csc {}'.format(cs_FileName))
    os.system('powershell.exe ./test_serial')


def checkFPGAOutputs(input_image, kernel_float, kernel_fixed, outputStreamFileName="fpgaOut.txt",expectedDim=518):

    """ 
    This function validates the FPGA Outputs via: 
        - Visual Inspection through OpenCV 
        - Correlation computation using Scipy (to be implemented later)

    @param outputStreamFileName: string containing the bytes from the FPGA, default of "fpgaOut.txt"
    @param expectedDim: int - the expected dimension size for the image (image_dim + zero padded layers)
    @param input_image: NumPy matrix containing the preprocessed greyscale image sent to the FPGA
    @param kernel_float: Numpy matrix containing double-precision kernel data
    @param kernel_fixed: numpy matrix containing fixed-point int8 kernel data
    """
    valid_double = correlate2d(input_image,kernel_float,boundary='same')
    valid_fixed = correlate2d(input_image,kernel_fixed,boundary='same')

    garbage = []
    
    expectedSize = expectedDim**2

    with open(outputStreamFileName,'r') as f:
        while True: 
            x = f.readline()
            y = f.readline()
            if not y: break

            # Remove the endline characters 
            x = x.rstrip('\n')
            y = y.rstrip('\n')

            garbage.append(x+y)

    # Casting to uint8 for now, cv2 crashes with int16
    fpgaOut = np.array(garbage,dtype=np.uint8)
    buf = np.zeros(shape=[expectedSize],dtype=np.uint8)

    # Adjust fpgaOut shape based on fpgaOut size
    if (fpgaOut.shape[0] < expectedSize):
        buf[0:fpgaOut.shape[0]] = fpgaOut
    elif (fpgaOut.shape[0] > expectedSize):
        buf[0:expectedSize] = fpgaOut[0:expectedSize]
    else:
        buf = fpgaOut

    # Reshape into image sizes 
    buf = np.reshape(buf, [expectedDim,expectedDim])

    # Remove the zero padded layers from buf
    num_pad_layers = int(kernel.shape[0]/2)
    buf = buf[num_pad_layers:-num_pad_layers, num_pad_layers:-num_pad_layers]

    #Show output through OpenCV
    cv2.imshow('Expected Output',valid_double)
    cv2.waitKey(0)

    cv2.imshow('Recieved Output',buf)
    cv2.waitKey(0)

    # Compute the error between the FPGA Results and the fixed point convolution

    err_double = np.absolute(buf-valid_double)
    err_fixed = np.absolute(buf-valid_fixed)

    print("Error between FPGA Result and double-precision convolution: {}".format(np.linalg.norm(err_double)))
    print("Error between FPGA results and fixed-point convolution: {}".format(np.linalg.norm(err_fixed)))    

    return (err_double, err_fixed)