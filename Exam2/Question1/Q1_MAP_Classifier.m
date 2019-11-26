function label = Q1_MAP_Classifier(x,mu,sigma,prior,N)

label = zeros(1,length(x));
MAP_result = zeros(N,length(x));
for n = 1:N
    MAP_result(n,:) = prior(n).*evalGaussian(x,mu(:,n),sigma(:,:,n));
end
[~,label] = max(MAP_result,[],1);
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
