function c = RiskMinimization(mu1,Sigma1,mu2,Sigma2,N,n)
%RISKMINIMIZATION Summary of this function goes here
%   Detailed explanation goes here
mu(:,1) = mu1;
mu(:,2) = mu2;
Sigma(:,:,1) = Sigma1;
Sigma(:,:,2) = Sigma2;
p1 = [0.5,0.5];% class priors for labels 0 and 1 respectively
p2 = [0.05,0.95];% class priors for labels 0 and 1 respectively
% generate the label of each sample 
label = rand(1,N) >= 0.5; 
% number of samples from each class
Nc = [length(find(label==0)),length(find(label==1))];
x = zeros(2,N);

for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
%Origin data
figure(n),
subplot(2,2,1);
plot(x(1,label==0),x(2,label==0),'bo',x(1,label==1),x(2,label==1),'ro');
legend('L = 1','L = 2','Location','NorthWest'); 
title('Data and their true labels');
xlabel('x_1'), ylabel('x_2');

%After classified
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*p1(2)/p1(1);%discriminant threshold
discriminantScore1 = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision1 = (discriminantScore1 >= log(gamma));

ind00 = find(decision1==0 & label==0); 
ind10 = find(decision1==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision1==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision1==1 & label==1); 
Pe1 = [p10,p01]*Nc'/N % probability of error
subplot(2,2,3);
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,

gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*p2(2)/p2(1);%discriminant threshold
discriminantScore2 = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision2 = (discriminantScore2 >= log(gamma));

ind2_00 = find(decision2==0 & label==0); 
ind2_10 = find(decision2==1 & label==0); p2_10 = length(ind2_10)/Nc(1);% probability of false positive
ind2_01 = find(decision2==0 & label==1); p2_01 = length(ind2_01)/Nc(2);% probability of false negative
ind2_11 = find(decision2==1 & label==1); 
Pe2 = [p2_10,p2_01]*Nc'/N % probability of error
subplot(2,2,4);
plot(x(1,ind2_00),x(2,ind2_00),'og'); hold on,
plot(x(1,ind2_10),x(2,ind2_10),'or'); hold on,
plot(x(1,ind2_01),x(2,ind2_01),'+r'); hold on,
plot(x(1,ind2_11),x(2,ind2_11),'+g'); hold on,
axis equal,

%Fisher LDA
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
% eig(A) returns diagonal matrix D of eigenvalues 
% and matrix V whose columns are the corresponding right eigenvectors,
% so that A*V = V*D
[V,D] = eig(inv(Sw)*Sb); %Sb * W = lambda * Sw * W, ie w is a generalized eigenvector of (Sw,Sb)
% diag() returns a square diagonal matrix with the elements of vector v on the main diagonal.
% [B,I] = sort(___) also returns a collection of index vectors for any of the previous syntaxes. 
% I is the same size as A and describes the arrangement of the elements of A into B along the sorted dimension. 
% For example, if A is a vector, then B = A(I).
[~,ind] = sort(diag(D),'descend');% extract eigenvalue and sort by descend
wLDA = V(:,ind(1)); % Fisher LDA projection vector 
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
% Y = sign(x) returns an array Y the same size as x, where each element of Y is:
% 1 if the corresponding element of x is greater than 0.
% 0 if the corresponding element of x equals 0.
% -1 if the corresponding element of x is less than 0.
% x./abs(x) if x is complex.
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures Label1 falls on the right side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
subplot(2,2,2)
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o', yLDA(find(label==1)),zeros(1,Nc(2)),'+'), 
axis equal,
legend('L = 1','L = 2'), 
title('Fisher LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
decisionLDA = (yLDA >= 0);




end

