N = 10;
gamma_noise = 0.1;%sigma of noise
Sigma_noise = gamma_noise^2; %sigma of noise

N_Gamma = 11;
L = zeros(100,N_Gamma);
Gamma = logspace(-10,10,N_Gamma);%sigma of w
for i = 1:11
%generate w_true based on different gamma
mu_true = [0;0;0;0];
gamma = Gamma(i);
Sigma_true = gamma^2 * eye(4);
%%w_true = mvnrnd(mu_true,Sigma_true)'
a = 0.01+rand(1);
b = -1+2*rand(1);
c = -1+2*rand(1);
d = -1+2*rand(1);
w_true = [a;-a*(b+c+d);a*(b*c+b*d+c*d);-a*b*c*d];
% 100 experiments
for j = 1:100
%generate x
x = -1+2*rand(1,N);
x = [x.^3 ; x.^2 ; x ; ones(1,N)];
%generate noise v
noise = mvnrnd(0,Sigma_noise,N);
%calculate y
y = x'*w_true+noise;
%get w_MAP
w_MAP = MAP(x',y',Sigma_noise*eye(N),Sigma_true)';
L(j,i) = norm(w_true-w_MAP,2)^2;
end
end
L = sort(L,1);
figure(1)
plot(log10(Gamma),log10(L(1,:)),"-b"),hold on,
plot(log10(Gamma),log10(L(25,:)),"-r"),hold on,
plot(log10(Gamma),log10(L(50,:)),"-g"),hold on,
plot(log10(Gamma),log10(L(75,:)),"-m"),hold on,
plot(log10(Gamma),log10(L(100,:)),"-c"),
xlabel('log_{10}(gamma)'), ylabel('log_{10}(L2 Distance between w_{true} and w_{MAP})'), 
legend("Minimum of SE","25% of SE","Median of SE","75% of SE","Maximum of SE","location","Northwest");
 
function w = MAP(input,output,Sigma1,Sigma2)
nume = output*inv(Sigma1)*input;
deno = input'*inv(Sigma1)*input+inv(Sigma2);
w = nume*inv(deno);
end




