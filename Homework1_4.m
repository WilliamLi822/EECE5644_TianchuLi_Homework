x=-3:0.01:3;
px1=Gaussian(x,0,1);
px2=Gaussian(x,1,2);
x0=sqrt(2*log(2)+2)-1;
x1=-sqrt(2*log(2)+2)-1;
plot(x,px1,'k',x,px2,'r',[x0,x0],[0,0.45],[x1,x1],[0,0.45]);
title('Plot of Gaussian Distribution');
xlabel('x'),ylabel('p(x|L=i)');
legend('p(x|L=1) \mu=0, \sigma^2=1','p(x|L=2) \mu=1, \sigma^2=2');

[X,Y] = meshgrid(1:0.5:10,1:5);
Z = sin(X) + cos(Y);
C = X.*Y;
surf(X,Y,Z,C)
size(X)
size(Y)
size(Z)

function px=Gaussian(x,Mu,Sigma)
C = (sqrt(2*pi*Sigma))^(-1);
E = -1/2*(x-Mu).^2/Sigma;
px = C*exp(E);
end
