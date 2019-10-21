function ClassIndex = Classifier1_1(data, mu, sigma, nSamples, prior, ClassIndex)

discriminatScore = zeros(3,nSamples);
for l = 1:3
    discriminatScore(l,:) = evalGaussian(data,mu(:,l),sigma(:,:,l))*prior(l);
    prior(l)
end
ClassTmp=max(discriminatScore,[],1);
for i = 1:nSamples
    [ClassIndex(2,i),~]= find(discriminatScore == ClassTmp(i));
end
end

