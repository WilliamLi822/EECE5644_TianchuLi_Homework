%produces N samples of independent and identically distributed (iid)
%n-dimensional random vectors {x1,x2,...,xN} drawn from N(?,?)

Nx=input('please enter the number of samples: ');
nx=input('please enter the number of dimensions of x: ');
flag=input('use your own mu and sigma(1) or generate them random(2): ');
x=-3+6*rand(nx,Nx);

%Gnerate random mu and sigma
if flag==2
    MuX=rand(nx,1);
    %Make sure that Sigma is a positive definite matrix
    D = diag(rand(nx,1));
    U = orth(rand(nx,nx));
    SigmaX = U' * D * U;
    %SigmaX=rand(nx,nx);
    %SigmaX(:,:) = tril(SigmaX(:,:),-1)+triu(SigmaX(:,:)',0);
end

Px=GaussianDistribution(x,MuX,SigmaX);

%Draw the figure of iid when n=1, 2
if nx==2
%Interpolation for surf() to draw the figure  
figure(1)
[X,Y,Z] = griddata(x(1,:),x(2,:),Px,linspace(min(x(1,:)),max(x(1,:)))',linspace(min(x(2,:)),max(x(2,:))),'nearest');
surf(X,Y,Z);
title('Figure of 2-dimension Gaussian Distribution');
elseif nx==1
plot(x,Px,'r+')
title('Figure of 1-dimension Gaussian Distribution');
xlabel('x'),ylabel('p(x)');
legend(['p(x) \mu=',num2str(MuX), ' \sigma^2=',num2str(SigmaX)]);
end

% Evaluates the Gaussian pdf N(mu,Sigma) at each sample of X
function G=GaussianDistribution(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma(:,:)))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
G = C*exp(E);
end