clc, clear, close all
rng(50)

pod = load('podaciCas03.txt');

ulaz = pod(:, 1:2);
izlaz = pod(:, 3);

%% One-hot encoding
izlazOH = zeros(length(izlaz), 3);
izlazOH(izlaz == 1, 1) = 1;
izlazOH(izlaz == 2, 2) = 1;
izlazOH(izlaz == 3, 3) = 1;

%% Podela na klase
K1 = ulaz(izlaz == 1, :);
K2 = ulaz(izlaz == 2, :);
K3 = ulaz(izlaz == 3, :);

figure, hold all
plot(K1(:, 1), K1(:, 2), 'o')
plot(K2(:, 1), K2(:, 2), '*')
plot(K3(:, 1), K3(:, 2), 'd')

%% Pripremanje podataka za format koji prima NM
ulaz = ulaz';
izlazOH = izlazOH';

%% Podela na trening i test skup
N = length(izlaz);
ind = randperm(N);
indTrening = ind(1 : 0.8*N);
indTest = ind(0.8*N+1 : N);

ulazTrening = ulaz(:, indTrening);
izlazTrening = izlazOH(:, indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = izlazOH(:, indTest);

%% Kreiranje NM
slojevi = [40, 30, 20, 10];
net = patternnet(slojevi);

net.divideFcn = '';
% net.divideFcn = 'dividerand';
% net.divideParam.trainRatio = 0.8;
% net.divideParam.testRatio = 0;
% net.divideParam.valRatio = 0.2;

net.performParam.regularization = 0.95;

for i = 1 : length(slojevi)
    net.layers{i}.transferFcn = 'tansig';
end
net.layers{i+1}.transferFcn = 'softmax';

net.trainParam.epochs = 2000;
net.trainParam.goal = 1e-5;
net.trainParam.min_grad = 1e-6;
net.trainParam.max_fail = 100;

%% Treniranje NM
net = train(net, ulazTrening, izlazTrening);

%% Performanse NM
pred = sim(net, ulazTest);
figure, plotconfusion(izlazTest, pred);

%% Granica odlucivanja
Ntest = 500;
ulazGO = [];
x1 = linspace(-4, 8, Ntest);
x2 = linspace(-4, 8, Ntest);

for x11 = x1
    pom = [x11*ones(1, Ntest); x2];
    ulazGO = [ulazGO, pom];
end

predGO = sim(net, ulazGO);
[vr, klasa] = max(predGO);

K1go = ulazGO(:, klasa == 1);
K2go = ulazGO(:, klasa == 2);
K3go = ulazGO(:, klasa == 3);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(:, 1), K1(:, 2), 'bo')
plot(K2(:, 1), K2(:, 2), 'r*')
plot(K3(:, 1), K3(:, 2), 'yd')
