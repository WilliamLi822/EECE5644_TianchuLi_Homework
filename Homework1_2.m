
%plot of  log-likelihood-ratio function
x=-10:0.1:10;
lx=abs(x-1)/2-abs(x)+log(2);
plot(x,lx);
title('Plot of Log-likelihood-ratio Function')
xlabel('x'), ylabel('l(x)')