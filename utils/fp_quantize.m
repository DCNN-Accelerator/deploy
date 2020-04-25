%%% 

% This function quantizes the double-precision kernel and image into the designated fixed-point datatypes for images
% and kernels 

% The datatypes are hardcoded as of 4/18, but future revisions should include parameterized decimal precision

% Kernel dtype - Q0.7
% Image  dtype - UQ8.0 (unsigned 8-bit integers)

%%%



function [quant_kernel, quant_img] = fp_quantize(kernel, image)

    % Reshape and transpose
    kernel = reshape(kernel,[7,7]);
    image = reshape(image,[518,518]);

    % Quantize to fixed-point using MATLAB's Fixed Point Designer
    % fi() creates a MATLAB fixed point object, which performs quantization based on the word length/fraction length parameters
    fp_kernel = fi(kernel, 1, 8, 7); 
    fp_image = fi(image, 0, 8, 0);
    
    % Return the 8-bit int values for Python processing
    quant_kernel = fp_kernel.int; 
    quant_img  = fp_image.int;

end