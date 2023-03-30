% deploy cnn_3 onto hardware
% load a pre-trained network
clear;

frameNumberLimit = 15;

net = load("Results\2023-03-30-10_25-net.mat").net;

% path for tools
hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', 'C:\Xilinx\Vivado\2020.2\bin\vivado.bat');

% target using JTAG connection
hTarget = dlhdl.Target('Xilinx', 'Interface', 'JTAG');

% use zcu111_single, this is the RFSoC board
hW = dlhdl.Workflow('Network', net, 'Bitstream', 'zcu111_single', 'Target', hTarget);

% compile hardware
% make sure that hardware normalization is off,
% use the normalization in input layer
dn = compile(hW, 'InputFrameNumberLimit', frameNumberLimit, 'HardwareNormalization', 'off')

% deplay the hardware
deploy(hW)

%
%
% Not yet implemented
%
% Include code for:
%   Before interfacing with FPGA
%       - Preprocess each image using dithering
%   Methodology for learning
%       - Place image with MATLAB
%       - Use source and DLP to display image, forward propgate using
%         optical elements
%       - Capture using CCD
%       - Perform flattening, softmax and classification layer in hardware
%       - Use hW object to backpropagate and retrieve learned kernel
%       - Update with new kernel, repeat
%
%   Prediction
%       - Place image with MATLAB
%       - Use source and DLP to display image, forward propagate using
%         optical elements
%       - Capture using CCD
%       - Perform flattening, softmax and classification layer in hardware
%       - Reteive predicted labels

