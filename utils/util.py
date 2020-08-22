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
import ctypes as ct


def image_rescale(image_path,dim):

    """
    Takes an input image path, and converts it to greyscale and the user dimensions (has to be square)
    @param image_path: string path containing the image file location
    """

    img = cv2.imread(image_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (dim,dim), interpolation=cv2.INTER_AREA)

def generate_sobel(kernel_size):

    """
    Creates a square Sobel Edge-Detection filter to be used for image filtering on the FPGA
    
    @param kernel_size: odd filter dimension 
    """
    kernel = np.zeros([kernel_size,kernel_size])
    kernel[0:int(kernel_size/2),:] = 1
    kernel[int(kernel_size/2)+1:kernel_size,:] = -1

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
    eng.addpath('C:/Users/hkhaj/Documents/senior-project/deploy/utils/')

    # MATLAB is dumb and can't process ndarrays directly, using a file-based workaround instead
    np.savetxt('kernel.txt',kernel)
    np.savetxt('image.txt',image)

    image_matlab  = eng.importdata('image.txt')
    kernel_matlab = eng.importdata('kernel.txt')
    
    fp_kernel, fp_image = eng.fp_quantize(kernel_matlab, image_matlab,nargout=2)

    # Cast back to nparrays 
    fp_image = np.asarray(fp_image, dtype=np.uint8)
    fp_kernel = np.asarray(fp_kernel,dtype=np.uint8)

    return [fp_kernel, fp_image]

def createUARTStream(image,kernel,streamFileName):

    """
    Creates a 1D stream containing kernel and image data and writes to a file
    Requires int8 quantized data
    @param image: numpy array containing image data
    @param kernel: numpy array containing kernel data
    @param streamFileName: string specifiying the filename that the stream should be written to 
    """

    # flatten and concatenate
    stream = np.concatenate((np.ndarray.flatten(kernel),np.ndarray.flatten(image)))
    
    x = np.asarray(stream,dtype=np.uint8)
    np.savetxt(streamFileName,x,fmt='%u')

def zeroPad(image,kernel_size):

    """
    Takes the image, and zero pads it according to the kernel size
    @param image: nparray containing image data
    @param kernel_size: int containing the square size of the kernel
    """
    num_pad_zeros = int(kernel_size/2)
    return np.pad(image,((num_pad_zeros,num_pad_zeros),(num_pad_zeros,num_pad_zeros)),mode='constant')

def twosComp (hex_str, num_bits=16):
    """
    Converts a hex string into a signed 16-bit integer
    
    @param hex_str: string containing the hexadecmimal value to be converted ex: '/FFAB'
    @param num_bits: int specifying the base of the result (default is 16)
    """

    value = int(hex_str,16)
    if value & (1 << (num_bits-1)):
        value -= (1<<num_bits)
    
    return value 



def runFPGAConvolution(inputStreamFileName="uart_input_bytes.txt", cs_FileName="test_serial.cs", outputStreamFileName="fpgaOut.txt"): 

    """
    This function invokes the C# program to send/recieve convolution data from the FPGA 
    Given a stream path, this function compiles the C# program and invokes the executable
    
    This method will fail if the C# compiler is not part of the System Path
     
        @param: inputStreamFileName: a path to the file containing the 8-bit bytes to be streamed to the FPGA
        @param: outputStreamFileName: a path to the file containing the FPGA outputs
        @param: cs_FileName: a string designating the C# file to be compiled (should be testSerial.cs by default)
    Notes: 
        was having trouble getting C# cmd-line args to work so for now all the output files are "fpgaOut.txt", 
        this method ignores the outputStreamFileName param for now
    """
    os.system('powershell.exe csc {}'.format(cs_FileName))
    os.system('C:/Users/hkhaj/Documents/senior-project/deploy/test_serial.exe')

def checkFPGAOutputs(input_image, kernel_float, kernel_fixed=None, outputStreamFileName="fpgaOut.txt",expectedDim=518):

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

    valid_double = correlate2d(input_image,kernel_float,'same')
    np.savetxt('expected.txt',np.ndarray.flatten(valid_double))
    
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
            
            #convert hex chars to int 
            garbage_str = x+y
            garbage_str = garbage_str.replace(" ","")
            garbage_int = twosComp(garbage_str)

            garbage.append(garbage_int)

    # Casting to uint8 for now, cv2 crashes with int16
    fpgaOut = np.asarray(garbage,dtype=np.int16)
    buf = np.zeros(shape=[expectedSize],dtype=np.int16)

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
    num_pad_layers = int(kernel_float.shape[0]/2)
    buf = buf[num_pad_layers:-num_pad_layers, num_pad_layers:-num_pad_layers]

    #Show output through OpenCV
    buf_normalized = cv2.normalize(buf,dst=None,alpha=-32768,beta=32767,norm_type=cv2.NORM_MINMAX)
    valid_normalized = cv2.normalize(valid_double.astype(np.int16),dst=None,alpha=-32768,beta=32767,norm_type=cv2.NORM_MINMAX)

    cv2.imshow('Original',input_image)
    cv2.waitKey(0)

    cv2.imshow('Expected Output',valid_normalized)
    cv2.waitKey(0)

    cv2.imshow('Recieved Output',buf_normalized)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    cv2.imwrite('expected.jpeg',~valid_normalized)
    cv2.imwrite('recieved.jpeg', ~buf_normalized)
    cv2.imwrite('grey_input.jpeg', input_image)
    return (buf)