clear
A1 = load('SAM1001_Ch1.dat');
B1 = load('SAM1001_Ch2.dat');
dt = 4e-6;
t = (0:length(A1)-1)*dt;
N1 = 188;
A1=circshift(A1,-N1);
B1=circshift(B1,-N1);
AA1=reshape(A1,250,40);
BB1=reshape(B1,250,40);
Aave1=mean(AA1,2);
Bave1=mean(BB1,2);

figure;
plot(t(1:250)*1e6,Aave1*16,t(1:250)*1e6,Bave1)

A2 = load('SAM1002_Ch1.dat');
B2 = load('SAM1002_Ch2.dat');
dt = 4e-6;
t = (0:length(A2)-1)*dt;
N2 = 186;
A2=circshift(A2,-N2);
B2=circshift(B2,-N2);
AA2=reshape(A2,250,40);
BB2=reshape(B2,250,40);
Aave2=mean(AA2,2);
Bave2=mean(BB2,2);
% plot(t(1:250)*1e6,Aave1*16,t(1:250)*1e6,Bave1,t(1:250)*1e6,Aave2*16,t(1:250)*1e6,Bave2)

figure;
plot(t(1:250)*1e6,(Aave1+Aave2)*8,t(1:250)*1e6,(Bave1+Bave2)/2)
