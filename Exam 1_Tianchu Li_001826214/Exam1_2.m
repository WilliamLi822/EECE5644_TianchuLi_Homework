
generateContour(1);
generateContour(2);
generateContour(3);
generateContour(4);

function generateContour(K)
sigma_x = 0.25;
sigma_y = 0.25;
sigma_noise = 0.3;
%%position_true = mvnrnd(mu_true,sigma_true)';
position_true = [0;0];

%generate K reference landmark
Angles = zeros(1,K);
startAngle = pi/K*rand(1);
for i = 1:K
    Angles(i) =  startAngle;
    startAngle = startAngle+2*pi/K;
end
landmark = [cos(Angles);sin(Angles)]

r = generateR(K,landmark,sigma_noise,position_true);
while size(find(r<0))>0
    r = generateR(K,sigma_noise,position_true);
end

x = -2:0.01:2;
y = -2:0.01:2;
for i = 1:size(x,2)
for j = 1:size(x,2)
z(i,j) = x(i)^2/sigma_x^2+y(j)^2/sigma_y^2 + sum((r-norm(landmark-repmat([x(i);y(j)],1,K),2)).^2)/sigma_noise^2;
end
end
figure(1),
subplot(2,2,K);
plot(position_true(1),position_true(2),"+r");hold on,
plot(landmark(1,:),landmark(2,:),"og");
contour(x,y,z);hold on,
legend("True position","Landmarks");


end

function r = generateR(K,landmark,sigma_noise,position_true)
%generate noise
noise = mvnrnd(0,sigma_noise,K);
%calculate dTi
dTi = norm(landmark-repmat(position_true,1,K),2);
%calculate ri
r = dTi+noise;
end
