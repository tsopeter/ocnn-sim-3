% Define physical parameters
wavelength = 0.5e-3; % wavelength of the wavefield in meters
z = 1e-4; % propagation distance in meters
Nx = 1024; % number of pixels in x direction
Ny = 1024; % number of pixels in y direction
Lx = 1e-3; % size of computational window in x direction in meters
Ly = 1e-3; % size of computational window in y direction in meters

% Define spatial coordinates
dx = Lx/Nx; % spatial step size in x direction in meters
dy = Ly/Ny; % spatial step size in y direction in meters
x = -Lx/2:dx:Lx/2;
y = -Ly/2:dy:Ly/2;
[X,Y] = meshgrid(x,y);

% Define input wavefield
u0 = exp(-((X.^2 + Y.^2)/(2*(0.1e-3)^2))); % Gaussian beam

% Perform Fourier transform of input wavefield
U0 = fftshift(fft2(u0));

% Define kx and ky coordinates
kx = 2*pi*linspace(-1/(2*dx), 1/(2*dx), Nx+1);
ky = 2*pi*linspace(-1/(2*dy), 1/(2*dy), Ny+1);
[Kx,Ky] = meshgrid(kx,ky);

% Define propagation kernel
k = 2*pi/wavelength;
H = exp(1i*k*z*sqrt(1 - (wavelength*Kx).^2 - (wavelength*Ky).^2));

% Apply propagation kernel in Fourier domain
U1 = U0.*H;

% Perform inverse Fourier transform to obtain output wavefield
u1 = ifft2(ifftshift(U1));

% Plot input and output wavefields
figure;
subplot(1,2,1);
imagesc(x*1e3,y*1e3,abs(u0).^2);
xlabel('x (mm)');
ylabel('y (mm)');
title('Input Wavefield');
subplot(1,2,2);
imagesc(x*1e3,y*1e3,abs(u1).^2);
xlabel('x (mm)');
ylabel('y (mm)');
title('Output Wavefield');
colorbar();


