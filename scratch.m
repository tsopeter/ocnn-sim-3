parameters_4;
addpath(genpath('SAM data'));
[P0, D0] = SAM_polyn_mW();

x    = linspace(0, 12, 100);
y    = polyval(P0, x);
dydx = polyval(D0, x);
xnormal = 8.708;
ynormal = 1.2756;

%checkLayer(CustomFFT1Layer('no_name', randn(32,1)), {[32 1]});
%checkLayer(CustomFastPropagationLayer('no_name', Nx, Ny, nx, ny, d1, wv), {[Nx, Ny]});
%checkLayer(CustomAmplitudeKernelLayer('no_name', randn(32)), {[32 32]});
%checkLayer(CustomNonlinearLayer('no_name', lvalue, sx, sy, sc, sz), {[Nx, Ny]});
%checkLayer(CustomFFT2PropagationLayer('no_name', Nx, Ny, nx, ny, d1, wv), {[Nx, Ny]});
%checkLayer(CustomDLPEndLayer('no_name', Nx, Ny, lvalue), {[Nx, Ny]});
checkLayer(CustomPolynomialNonLinearLayer('no_name', P0, D0, [32, 32, 1], xnormal, ynormal), {[32, 32]});

%d1 = 1e-9;
%H = angular_propagation(Nx, Ny, nx, ny, wv, d1);
%X = resize_normalize_extend(single(imread("TestImagesPNG/00001.png")), Nx, Ny, k, lvalue, 255);
%Y = ifft2(ifftshift(fftshift(fft2(X)) .* H));
%
%imagesc(real(Y.*conj(Y)));
%colorbar();