parameters_2;
%checkLayer(CustomFFT1Layer('no_name', randn(32,1)), {[32 1]});
%checkLayer(CustomFastPropagationLayer('no_name', Nx, Ny, nx, ny, d1, wv), {[Nx, Ny]});
%checkLayer(CustomAmplitudeKernelLayer('no_name', randn(32)), {[32 32]});
%checkLayer(CustomNonlinearLayer('no_name', lvalue, sx, sy, sc, sz), {[Nx, Ny]});
%checkLayer(CustomFFT2PropagationLayer('no_name', Nx, Ny, nx, ny, d1, wv), {[Nx, Ny]});
%checkLayer(CustomDLPEndLayer('no_name', Nx, Ny, lvalue), {[Nx, Ny]});

d1 = 1e-9;
H = angular_propagation(Nx, Ny, nx, ny, wv, d1);
X = resize_normalize_extend(single(imread("TestImagesPNG/00001.png")), Nx, Ny, k, lvalue, 255);
Y = ifft2(ifftshift(fftshift(fft2(X)) .* H));

imagesc(real(Y.*conj(Y)));
colorbar();