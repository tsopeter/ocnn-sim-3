function hB = plugin_board_PYNQ()
    hB = hdlcoder.Board;

    hB.BoardName = 'XUP PYNQ-Z2';

    hB.FPGAVendor  = 'Xilinx';
    hB.FPGAFamily  = 'Zynq';
    hB.FPGADevice  = 'xc7z020';
    hB.FPGAPackage = 'clg400c';
    hB.FPGASpeed   = '-1';

    % Tool Information
    hB.SupportedTool = {'Xilinx Vivado'};
    hB.JTAGChainPosition = 2;

    % Add Interfaces
    % Standard "External Port" Interface
    hB.addExternalPortInterface(...
        'IOPadConstraint', {'IOSTANDARD = LVCMOS33'});
    

end