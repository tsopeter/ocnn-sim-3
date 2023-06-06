x=linspace(-2,2,100);
y=0.5.*x.*(1+tanh((2/pi).*(x+0.044715*x.^3)));

figure;
plot(x,y);