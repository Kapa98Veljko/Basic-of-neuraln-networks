clc, clear, close all
rng(50)

%% Ucitaanje podataka
load podaciCas02.mat

ulaz = pod(1:2, :);
izlaz = pod(3, :);

K1 = ulaz(:, izlaz == 1);
K2 = ulaz(:, izlaz == 0);

figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')

%% Podela podataka na trening i test skup
N = length(izlaz);
ind = randperm(N);
ind_trening = ind(1 : 0.9*N);
ind_test = ind(0.9*N+1 : N);

ulaz_trening = ulaz(:, ind_trening);
izlaz_trening = izlaz(ind_trening);

ulaz_test = ulaz(:, ind_test);
izlaz_test = izlaz(ind_test);

%% Prikaz raspodele odbiraka na trening i test skup
figure, hold all
plot(ulaz_trening(1, :), ulaz_trening(2, :), 'o');
plot(ulaz_test(1, :), ulaz_test(2, :), '*');

%% Kreiranje neuralne mreze
net = patternnet([5 3]);

net.divideFcn = '';

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-3;
net.trainParam.min_grad = 1e-4;

net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';

%% Treniranje neuralne mreze
net = train(net, ulaz_trening, izlaz_trening);

%% Performanse neuralne mreze
pred = net(ulaz_test);
figure, plotconfusion(izlaz_test, pred);

pred = round(pred);
[cm, e] = confusionmat(izlazTest, pred);
cm = cm';

P = cm(2, 2)/(cm(2, 2) + cm(2, 1)); % preciznost
R = cm(2, 2)/(cm(2, 2) + cm(1, 2)); % osetljivost
F1 = 2*P*R/(P+R); % F1 skor
A = (cm(1, 1) + cm(2, 2))/sum(sum(cm)); % Tacnost

%% Granica odlucivanja
Ntest = 500;
x1Test = linspace(-10, 10, Ntest);
x2Test = linspace(-10, 10, Ntest);

ulaz_testGO = [];
for k = x2Test
    ulaz_testGO = [ulaz_testGO, [x1Test; k*ones(1, Ntest)]];
end

pred2 = net(ulaz_testGO);

prag = 0.5; % Gde zelimo da donesemo odluku
K1test = ulaz_testGO(:, pred2 >= prag);
K2test = ulaz_testGO(:, pred2 < 1-prag);

figure, hold all
plot(K1test(1, :), K1test(2, :), '.')
plot(K2test(1, :), K2test(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
