%%% 

% This function quantizes the double-precision kernel and image into the designated fixed-point datatypes for images
% and kernels 

% The datatypes are hardcoded as of 4/18, but future revisions should include parameterized decimal precision

% Kernel dtype - Q0.7
% Image  dtype - UQ8.0 (unsigned 8-bit integers)

%%%



function [quant_kernel, quant_img] = fp_quantize(d_kernel, d_image)

    fp_kernel = fi(d_kernel, 1, 8, 7); 
    fp_image = fi(d_image, 0, 8, 0);
    
    % Return the 8-bit int values for Python processing
    quant_kernel = fp_kernel.int; 
    quant_img  = fp_image.int;

end