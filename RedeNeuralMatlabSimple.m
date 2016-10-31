%{
***************************************************************************
* Rede Neural por Backpropagation (BP) 
* Marcio Pinto Pereira - julho de 2016
* Programado em Matlab R2016a 
* Licenciado sob CC-BY-SA
***************************************************************************
%}

load('RedeNeuralMatlabSimple.mat');
x = entrada1';
t = quality';
erro_min=1e99;
treino=zeros(50,100);
valida=zeros(50,100);
teste=zeros(50,100);

%while 1,
    for n=6:6,
        disp(n);
        for i=1:10;
            disp(i);
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = n;
net = fitnet(hiddenLayerSize,trainFcn);



% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';
%net.divideParam.trainRatio = 70/100;
%net.divideParam.valRatio = 15/100;
%net.divideParam.testRatio = 15/100;
[trainInd8,valInd8,testInd8] = divideint(1:18,.7,.15,.15);
[trainInd7,valInd7,testInd7] = divideint(19:40,.7,.15,.15);
[trainInd6,valInd6,testInd6] = divideint(41:62,.7,.15,.15);
[trainInd5,valInd5,testInd5] = divideint(63:83,.7,.15,.15);
[trainInd4,valInd4,testInd4] = divideint(84:110,.7,.15,.15);
[trainInd3,valInd3,testInd3] = divideint(111:120,.7,.15,.15);
net.divideFcn = 'divideind';
net.divideParam.trainInd=[trainInd8 trainInd7 trainInd6 trainInd5 trainInd4 trainInd3];
net.divideParam.valInd=[valInd8 valInd7 valInd6 valInd5 valInd4 valInd3];
net.divideParam.testInd=[testInd8 testInd7 testInd6 testInd5 testInd4 testInd3];
%[tr.trainInd,tr.valInd,tr.testInd] = divideind(120,trainInd,valInd,testInd);

% Train the Network
net.trainParam.showWindow=false;
[net,tr] = train(net,x,t);

%tr.perf, tr.vperf, tr.tperf tem os erros

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

treino(n,i)= tr.best_perf;
valida(n,i)= tr.best_vperf;
teste(n,i)= tr.best_tperf;
    
soma_erros=tr.best_perf + tr.best_vperf + tr.best_tperf;
if (soma_erros < erro_min )
    erro_min=soma_erros;
    best_tr=tr;
    best_net=net;
end;
        end;
    end;
disp(best_tr.best_tperf);
%end;
% View the Network
%view(net)
y = best_net(x);
e = gsubtract(t,y);
performance = perform(best_net,t,y);
% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(best_tr)
figure, plottrainstate(best_tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)
%{
for i=1:50, best_teste(i)=min(teste(i,:)); end;
figure;
grid('on');
plot(best_teste);
%}


