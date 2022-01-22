clc, clear, close all
rng(50)

%% Ucitavanje podataka
pod = readtable('podaciCas04.csv');

ulaz = [pod.x1, pod.x2];
izlaz = pod.d;

ulaz = ulaz';
izlaz = izlaz';

%% Prikaz raspdele odbiraka po klasama
figure
histogram(izlaz)

%% Podela na klase
K1 = ulaz(:, izlaz == 1);
K2 = ulaz(:, izlaz == 0);

figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')

%% Izdvajanje trening, test i val skupa za svaku prvu klasu
N1 = length(K1);
K1trening = K1(:, 1 : 0.7*N1);
K1test = K1(:, 0.7*N1+1 : 0.85*N1);
K1val = K1(:, 0.85*N1+1 : N1);

%% Izdvajanje trening, test i val skupa za svaku drugu klasu
N2 = length(K2);
K2trening = K2(:, 1 : 0.7*N2);
K2test = K2(:, 0.7*N2+1 : 0.85*N2);
K2val = K2(:, 0.85*N2+1 : N2);

%% Formiranje zajednockog trening, test i val skupa
ulazTrening = [K1trening, K2trening];
izlazTrening = [ones(1, 0.7*N1), zeros(1, 0.7*N2)];

ind = randperm(length(izlazTrening));
ulazTrening = ulazTrening(:, ind);
izlazTrening = izlazTrening(ind);

ulazTest = [K1test, K2test];
izlazTest = [ones(1, 0.15*N1), zeros(1, 0.15*N2)];

ulazVal = [K1val, K2val];
izlazVal = [ones(1, 0.15*N1), zeros(1, 0.15*N2)];

%% Formiranje skupa koji ce se prosledidi NM za treniranje (sadrzi validaciju)
ulazSve = [ulazTrening, ulazVal];
izlazSve = [izlazTrening, izlazVal];

%% Krosvalidacija
arhitektura = {[10, 5], [12, 6, 3], [4]};
Abest = 0;
F1best = 0;

for reg = [0.1, 0.5, 0.9]
    for w = [2, 5, 10]
        for lr = [0.5, 0.05, 0.005]
            for arh = length(arhitektura)
                rng(5)
                net = patternnet(arhitektura{arh});

                net.divideFcn = 'divideind';
                net.divideParam.trainInd = 1 : length(ulazTrening);
                net.divideParam.valInd = length(ulazTrening)+1 : length(ulazSve);
                net.divideParam.testInd = [];

                net.performParam.regularization = reg;

                net.trainFcn = 'traingd';

                net.trainParam.lr = lr;
                net.trainParam.epochs = 1000;
                net.trainParam.goal = 1e-4;
                net.trainParam.max_fail = 20;
                net.trainParam.showWindow = false;

                weight = ones(1, length(izlazSve));
                weight(izlazSve == 1) = w;

                [net, info] = train(net, ulazSve, izlazSve, [], [], weight);

                pred = sim(net, ulazVal);
                pred = round(pred);

                [~, cm] = confusion(izlazVal, pred);
                A = 100*sum(trace(cm))/sum(sum(cm));
                F1 = 2*cm(2, 2)/(cm(2, 1)+cm(1, 2)+2*cm(2, 2));

                disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ', F1 = ' num2str(F1)])
                disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])

                if F1 > F1best
                    F1best = F1;
                    Abest = A;
                    reg_best = reg;
                    w_best = w;
                    lr_best = lr;
                    arh_best = arhitektura{arh};
                    ep_best = info.best_epoch;
                end
            end
        end
    end
end

%%Treniranje NM sa optimalnim parametrima (na celom trening + val skupu)
net = patternnet(arh_best);

net.divideFcn = '';

net.performParam.regularization = reg_best;

net.trainFcn = 'traingd';

net.trainParam.lr = lr_best;

net.trainParam.epochs = ep_best;
net.trainParam.goal = 1e-4;

weight = ones(1, length(izlazSve));
weight(izlazSve == 1) = w_best;

[net, info] = train(net, ulazSve, izlazSve, [], [], weight);

%% Performanse NM
pred = sim(net, ulazTest);
figure, plotconfusion(izlazTest, pred);
