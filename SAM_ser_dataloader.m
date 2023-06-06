%
% load data
% P0    - polynomials fitted to data
% D0    - derivatives of polynomials fitted to data
function [P0, D0, t0] = SAM_ser_dataloader(n_poly)
    ch1   = load('SAM data-2/SAM1001_Ch1.dat');
    ch2   = load('SAM data-2/SAM1001_Ch2.dat');
    dt    = 4e-6;
    t     = (0:length(ch1)-1)*dt;
    n1    = 188;
    b1    = circshift(ch2, -n1);
    bb1   = reshape(b1, 250, 40);
    bave1 = mean(bb1, 2);

    ss = length(bave1); % remove like 5% from both sides
    cp = ceil(ss*0.05);
    s2 = bave1(cp:ss-cp);
    t2 = t(1:250)*1e6;
    t2 = t2(cp:ss-cp);

    % fit to polynomials
    P0 = polyfit(t2, s2, n_poly);
    D0 = polyder(P0);
    t0 = t2;

    %
    figure;
    plot(t2, polyval(P0, t2));
end