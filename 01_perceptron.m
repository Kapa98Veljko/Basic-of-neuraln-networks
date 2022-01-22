clc, clear, close all
rng(50)

%% Generesianje odbiraka
N = 500; % Broj odbiraka svake od klasa

% Generisanje odbiraka prve klase sa normlanom raspodelom
x1 = randn(1, N);
y1 = randn(1, N);
K1 = [x1; y1];

% Dodati generisanje odbiraka druge klase
x2 = randn(1, N) + 3;
y2 = randn(1, N) + 2;
K2 = [x2; y2];

%% Prikaz podataka
figure, hold all
plot(x1, y1, 'o');
plot(x2, y2, '*')

%% Kreiranje ulaza i izlaza perceptrona
% Ovde definisati matricu ulaza
ulaz = [K1, K2];

% Definisati vektor izlaza
izlaz = [ones(1, N), zeros(1, N)];

%% Obucavanje i testiranje
% Kreirati perceptron
per = perceptron('hardlim');

% Obuciti perceptron
per.trainParam.epochs = 100; % Maksimalan broj epoha treniranja
per.trainParam.goal = 1e-3; % Minimalna vrednost greske

view(per)

per = train(per, ulaz, izlaz);

% Testirati perceptron
pred = sim(per, ulaz);

% Prikazati matricu konfuzije
figure
plotconfusion(izlaz, pred)

%% Granica odlucivanja
Ntest = 500;
% Formirati ulazni vektor za testiranje
x1test = linspace(-4, 8, Ntest);
x2test = linspace(-3, 7, Ntest);
ulazTest = [];
for i = x2test
    ulazTest = [ulazTest, [x1test; i*ones(1, Ntest)]];
end

% Testirati obucen perceptron za formiran test skup
predTest = sim(per, ulazTest);
K1 = ulazTest(:, predTest == 1);
K2 = ulazTest(:, predTest == 0);

% Prikazati granicu odlucivanja
figure, hold all
plot(K1(1, :), K1(2, :), '.')
plot(K2(1, :), K2(2, :), '.')
plot(x1, y1, 'bo');
plot(x2, y2, 'ro')
