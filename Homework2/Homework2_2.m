n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu1 = [0;0]; 
mu2 = [3;3];
mu3 = [2;2];
Sigma1 = [1,0;0,1]; 
Sigma2 = [3,1;1,0.8];
Sigma3 = [2,0.5;0.5,1];
Sigma4 = [2,-1.9;-1.9,5];


RiskMinimization(mu1,Sigma1,mu2,Sigma1,N,1);
%RiskMinimization(mu1,Sigma2,mu2,Sigma2,N,2);
%RiskMinimization(mu1,Sigma3,mu3,Sigma4,N,3);

