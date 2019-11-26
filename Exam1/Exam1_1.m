m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(2,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    length(indices)
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), subplot(2,2,1),plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
legend('Class = 1','Class = 2','Class = 3','Location','SouthWest');
L=Classifier1_1(x, m, Sigma, N, classPriors, L);

indMAP1 = find(L(1,:)==1);
indMAP2 = find(L(1,:)==2);
indMAP3 = find(L(1,:)==3);

ConfusionMatrix = zeros(3,3);
for l = 1:3
    ConfusionMatrix(l,1) = length(find(L(2,indMAP1) == l));
    ConfusionMatrix(l,2) = length(find(L(2,indMAP2) == l));
    ConfusionMatrix(l,3) = length(find(L(2,indMAP3) == l));
end
indMAP11 = find(L(2,indMAP1) == 1);
indMAP12 = find(L(2,indMAP1) == 2);
indMAP13 = find(L(2,indMAP1) == 3);
indMAP21 = find(L(2,indMAP2) == 1);
indMAP22 = find(L(2,indMAP2) == 2);
indMAP23 = find(L(2,indMAP2) == 3);
indMAP31 = find(L(2,indMAP3) == 1);
indMAP32 = find(L(2,indMAP3) == 2);
indMAP33 = find(L(2,indMAP3) == 3);

ConfusionMatrix
error = length(indMAP12)+length(indMAP13)+length(indMAP21)+length(indMAP23)+length(indMAP31)+length(indMAP32)
pe = error/N

subplot(2,2,2),
plot(x(1,[indMAP11,indMAP22,indMAP33]),x(2,[indMAP11,indMAP22,indMAP33]),'.g'); axis equal, hold on,
plot(x(1,indMAP12),x(2,indMAP12),'+m'); axis equal, hold on,
plot(x(1,indMAP13),x(2,indMAP13),'ob'); axis equal;
legend('All Data','Decision Class = 2','Decision Class = 3','Location','SouthWest');
title('True Class 1');
subplot(2,2,3),
plot(x(1,[indMAP11,indMAP22,indMAP33]),x(2,[indMAP11,indMAP22,indMAP33]),'.g'); axis equal, hold on,
plot(x(1,indMAP21),x(2,indMAP21),'+m'); axis equal, hold on,
plot(x(1,indMAP23),x(2,indMAP23),'ob'); axis equal;
legend('All Data','Decision Class = 1','Decision Class = 3','Location','SouthWest');
title('True Class 2');
subplot(2,2,4),
plot(x(1,[indMAP11,indMAP22,indMAP33]),x(2,[indMAP11,indMAP22,indMAP33]),'.g'); axis equal, hold on,
plot(x(1,indMAP31),x(2,indMAP31),'+m'); axis equal, hold on,
plot(x(1,indMAP32),x(2,indMAP32),'ob'); axis equal;
legend('All Data','Decision Class = 1','Decision Class = 2','Location','SouthWest');
title('True Class 3');

