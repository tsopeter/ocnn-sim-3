parameters;
%checkLayer(CustomFFT1Layer('no_name', randn(32,1)), {[32 1]});
%checkLayer(CustomFastPropagationLayer('no_name', Nx, Ny, nx, ny, d1, wv), {[Nx, Ny]});
%checkLayer(CustomAmplitudeKernelLayer('no_name', randn(32)), {[32 32]});
checkLayer(CustomNonlinearLayer('no_name', lvalue, sx, sy, sc, sz), {[Nx, Ny]});