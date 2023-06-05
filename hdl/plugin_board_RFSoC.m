function hB = plugin_board_RFSoC()
    % Board Definition
    hB = hdlcoder.Board;

    hB.BoardName = 'Zynq UltraScale+ RFSoC ZCU111 Evaluation Kit';

    hB.FPGAVendor = 'Xilinx';
    hB.FPGAFamily = 'Zynq';
    hB.FPGADevice = '';

end