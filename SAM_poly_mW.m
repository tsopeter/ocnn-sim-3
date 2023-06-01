% Same as SAM_poly
% but assumes that the base value is in mW
% instead of W
function [P0, D0] = SAM_poly_mW()
    %
    % scale
    lm0  = load('SAM data/Linear_mirror.txt');
    a0   = 10.^(lm0/10);
    x0   = linspace(a0(1,1),a0(end,1),100);
    R0   = a0(:,2)./a0(:,1);
    Y0   = interp1(a0(:,1),R0,x0);
    YY0  = polyfit(x0,Y0,1);
    y0   = polyval(YY0,x0);
    scal = mean(y0);

    sm0 = load('SAM data/SAM_measured0.txt');
    a1  = 10.^(sm0(1:15,:)/10);
    R1  = a1(:,2)./a1(:,1);
    x1  = linspace(a1(1,1),a1(end,1),100);
    Y1  = interp1(a1(:,1),R1,x1);
    YY1 = polyfit(x1,Y1,5);

    P0  = YY1/scal;
    D0  = polyder(P0);

    %figure;
    %semilogx(x1,polyval(P0,x1));
end