% MATLAB script to check the RTL outputs 

% Change this parameter to the dimensions of the image 
img_size = 518; 

% Assuming a sq. feature map output, we allocate two bytes for each element
num_fm_elements = (img_size^2) ; 

% Fixed-Point arrays to hold output data
% valid_fp   = zeros (num_fm_elements,1);
rtl_fp     = zeros (num_fm_elements,1); 

% Open files to read in bytes
% valid_file = fopen('C:/Users/ryanm/Downloads/testing/BUILD/matlab/conv_valid_bytes.txt');

% change the path of this file to whatever the output file of the VHDL model is
rtl_file   = fopen('fpgaOut.txt'); 
chars = [];

for i = 1: num_fm_elements
    
    % read 2 lines at a time to get each byte from the rtl file
    str_1 = fgetl(rtl_file); 
    str_2 = fgetl(rtl_file); 
    data  = strcat(str_1, str_2); 
    
    chars = [chars, data];
    x = fi([],1, 16, 0); 
    x.hex = data; 
    rtl_fp(i) = x; 
    
%     %read each 16b value from the "conv_valid" file 
%     data = fgetl(valid_file); 
%     x.hex = data; 
%     
%     valid_fp(i) = x; 
   
end 


% disp (" Average error from theoretical fixed-point convolution and RTL results (error per byte)" ) ; 
% disp ( sum(sum( abs(rtl_fp - valid_fp) )) / num_fm_elements); 
% 
% fclose(valid_file); 
% fclose(rtl_file); 