clear

load SAM_measured0.txt  %1st measurement
a1 = 10.^(SAM_measured0(1:15,:)/10);
x1 = a1(1,1):0.01:a1(end,1);
Y1 = interp1(a1(:,1),a1(:,2),x1);
YY1 = polyfit(x1,Y1,5);
y1 = polyval(YY1,x1);
% plot(a1(:,1),a1(:,2),'.',x1,y1)

load SAM_measured1.txt  %2nd measurement 
a2 = 10.^(SAM_measured1(1:end-1,:)/10);
x2 = a2(1,1):0.01:a2(end,1);
Y2 = interp1(a2(:,1),a2(:,2),x2);
YY2 = polyfit(x2,Y2,8);
y2 = polyval(YY2,x2);
% plot(a2(:,1),a2(:,2),'.',x2,y2)

load Linear_mirror.txt %replace SAM with a regular mirror
a0 = 10.^(Linear_mirror/10);
x0 = a0(1,1):0.01:a0(end,1);
Y0 = interp1(a0(:,1),a0(:,2),x0);
YY0 = polyfit(x0,Y0,1);
y0 = polyval(YY0,x0);
% plot(a1(:,1),a1(:,2),'.',x1,y1,a0(:,1),a0(:,2)/2,'.',x0,y0/2)
% plot(a2(:,1),a2(:,2),'.',x2,y2,a0(:,1),a0(:,2)/2.2,'.',x0,y0/2.2)

plot(a1(:,1),a1(:,2),'.',x1,y1,a2(:,1),a2(:,2),'.',x2,y2,a0(:,1),a0(:,2)/2,'.',x0,y0/2)
axis([0 10 0 1.1])
grid
xlabel('Input power (mW)')
ylabel('Reflected power (mW)')
