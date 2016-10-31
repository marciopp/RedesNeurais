%{
***************************************************************************
* Rede Neural por Backpropagation (BP) 
* Marcio Pinto Pereira - julho de 2016
* Programado em Matlab R2016a 
* Licenciado sob CC-BY-SA
***************************************************************************
%}
clear all; close all;
load('Neural_Final_2.mat');
err=zeros(10,1);
err_teste=zeros(10,1);
err_valida=zeros(10,1);
percentual_acerto_otimo=0;
m_erro_n=zeros(15,100);
m_erro_teste_n=zeros(15,100);
m_erro_valida_n=zeros(15,100);
gap_anterior=1e99; val_fail=0;
% Parâmetros
for neuron=10:10,
for tenta=1:1,
    disp(tenta);
clear entrada;
clear saida;
clear teste;
clear valida;
clear erro;
neuronios_ocultos = neuron;
etapas = 1000;
% Entrada e Saída

percentual_acerto_atual=0;

%{
%Teste do XOR
entrada1=[0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    0 0; 0 1; 1 0; 1 1
    ];
saida1=[0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 ]';
%}


%load('RedWhite1Wine.mat');
%entrada1=[alcohol chlorides citricacid fixedacidity sulphates totalsulfurdioxide];
%entrada1=[alcohol chlorides citricacid fixedacidity sulphates totalsulfurdioxide volatileacidity density residualsugar pH freesulfurdioxide];


entrada1=[alcohol chlorides citricAcid sulphates totalSulfurDioxide];
saida1=quality;
%}

%{
%Teste do seno
x=[0:0.1:2*pi];
y1=  sin(x)+randn(1,length(x))/100;
y2= -sin(x)+randn(1,length(x))/100;
entrada1=x';
saida1=[ y2'];
%}

% Embaralhando conjuntos de treinamento, teste e validação
p=randperm(length(entrada1));
px=p(1:round(length(entrada1)*0.7));
pteste=p(round(length(entrada1)*0.7):round(length(entrada1)*0.85));
pvalida=p(round(length(entrada1)*0.85):length(entrada1));

% Normalizando pares de treinamento
for i=1:length(px)
entrada(i,:)=entrada1(px(i),:);
saida(i,:)=saida1(px(i),:);
end;
media_entrada = mean(entrada);
desvio_padrao_entrada = std(entrada);
for i=1:size(entrada,2)
    entrada(:,i) = (entrada(:,i) - media_entrada(1,i)) / desvio_padrao_entrada(1,i);
end;
media_saida = mean(saida);
desvio_padrao_saida = std(saida);
for i=1:size(saida,2)
    saida(:,i) = (saida(:,i) - media_saida(1,i)) / desvio_padrao_saida(1,i);
end;

% Normalizando pares de teste
for i=1:length(pteste)
teste(i,:)=entrada1(pteste(i),:);
saida_teste(i,:)=saida1(pteste(i),:);
end;
media_teste = mean(teste);
desvio_padrao_teste = std(teste);
for i=1:size(teste,2)
    teste(:,i) = (teste(:,i) - media_teste(1,i)) / desvio_padrao_teste(1,i);
end;
media_saida_teste = mean(saida_teste);
desvio_padrao_saida_teste = std(saida_teste);
for i=1:size(saida_teste,2)
    saida_teste(:,i) = (saida_teste(:,i) - media_saida_teste(1,i)) / desvio_padrao_saida_teste(1,i);
end;

% Normalizando pares de validação
for i=1:length(pvalida)
valida(i,:)=entrada1(pvalida(i),:);
saida_valida(i,:)=saida1(pvalida(i),:);
end;
media_valida = mean(valida);
desvio_padrao_valida = std(valida);
for i=1:size(valida,2)
    valida(:,i) = (valida(:,i) - media_valida(1,i)) / desvio_padrao_valida(1,i);
end;
media_saida_valida = mean(saida_valida);
desvio_padrao_saida_valida = std(saida_valida);
for i=1:size(saida_valida,2)
    saida_valida(:,i) = (saida_valida(:,i) - media_saida_valida(1,i)) / desvio_padrao_saida_valida(1,i);
end;

if size(entrada,1) ~= size(saida,1)
    disp('ERRO: pares entrada/saída com quantidades diferentes')
   return 
end

% Inserção do bias
pares_treinamento = size(entrada,1);
bias = ones(pares_treinamento,1);
entrada = [entrada bias];
pares_treinamento_teste = size(teste,1);
bias = ones(pares_treinamento_teste,1);
teste = [teste bias];
pares_treinamento_valida = size(valida,1);
bias = ones(pares_treinamento_valida,1);
valida = [valida bias];

% Verifica número de entradas e saídas
n_entradas  = size(entrada,2);
n_saidas = size(saida,2);
%Inicializa pesos
pesos_entrada_oculta  = (rand(n_entradas, neuronios_ocultos) - 0.5)/10;
pesos_oculta_saida    = (rand(n_saidas,neuronios_ocultos)    - 0.5)/10;
erromin=1e99; errmin=1e99;
%%
% Gráfico de observação das curvas
% Botão de parada
set(0, 'DefaultFigurePosition', [ 100 100 500 500 ]);
hstop = uicontrol('Style','PushButton','String','Parar', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;
%Botão para zerar pesos
hreset = uicontrol('Style','PushButton','String','Zerar Pesos', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;
% Botão de taxa de aprendizado
hlr = uicontrol('Style','slider','value',.1,'Min',.01,'Max',1,'SliderStep',[0.01 0.1],'Position', get(hreset,'position')+[75 0 100 0]);

%%
delta_HO=zeros(neuron,1);
delta_IH=zeros(n_entradas,neuron);
for iter = 1:etapas
    alr = 0.01;
    blr=alr;
        %Seleção aleatória dos pares_treinamento em cada etapa
    for j = 1:pares_treinamento
        npar = round((rand * pares_treinamento) + 0.5);
        if npar > pares_treinamento
            npar = pares_treinamento;
        elseif npar < 1
            npar = 1;    
        end
        par_atual_entrada = entrada(npar,:);
        par_atual_saida = saida(npar,:); 
        
        npar_teste = round((rand * pares_treinamento_teste) + 0.5);
        if npar_teste > pares_treinamento_teste
            npar_teste = pares_treinamento_teste;
        elseif npar_teste < 1
            npar_teste = 1;    
        end
        par_atual_teste = teste(npar_teste,:);
        par_atual_saida_teste = saida_teste(npar_teste,:); 

        npar_valida = round((rand * pares_treinamento_valida) + 0.5);
        if npar_valida > pares_treinamento_valida
            npar_valida = pares_treinamento_valida;
        elseif npar_valida < 1
            npar_valida = 1;    
        end
        par_atual_valida = valida(npar_valida,:);
        par_atual_saida_valida = saida_valida(npar_valida,:); 
        
        % passo forward propagation
        % camada intermediaria tgh
        u_int = par_atual_entrada*pesos_entrada_oculta;
        v_int = tanh(u_int)'; %ok
        % camada saida linear
        u_ext = v_int'*pesos_oculta_saida'; % <--- saida !!!
        % erro
        epsilon = par_atual_saida - u_ext; %ok
        
        % Teste
        % passo forward propagation
        % camada intermediaria tgh
        u_int_teste = par_atual_teste*pesos_entrada_oculta;
        v_int_teste = tanh(u_int_teste)'; %ok
        % camada saida linear
        u_ext_teste = v_int_teste'*pesos_oculta_saida'; % <--- saida !!!
        % erro
        epsilon_teste = par_atual_saida_teste - u_ext_teste;

        %Validação
        % passo forward propagation
        % camada intermediaria tgh
        u_int_valida = par_atual_valida*pesos_entrada_oculta;
        v_int_valida = tanh(u_int_valida)'; %ok
        % camada saida linear
        u_ext_valida = v_int_valida'*pesos_oculta_saida'; % <--- saida !!!
        % erro
        epsilon_valida = par_atual_saida_valida - u_ext_valida;
                
        erro(:,j) = sum(epsilon.^2); %erro rms
        erro_teste(:,j) = sum(epsilon_teste.^2);
        erro_valida(:,j) = sum(epsilon_valida.^2);
        
        if (erro(:,j) < erromin)
            p_o_s_min=pesos_oculta_saida;
            p_e_o_min=pesos_entrada_oculta;
            jmin=j;
            erromin=erro(:,j);
        end;
        % passo back propagation
        % camada saida linear - phi_linha = 1;
        phi_linha = ones(neuronios_ocultos,1); % phi_linha = (1-v_ext.^2)
        delta = phi_linha * epsilon;
        for i=1:n_saidas
            delta_HO (:,i)= delta(:,i).* v_int;
        end;
        delta_HO = 2* blr * delta_HO;
        pesos_oculta_saida = pesos_oculta_saida + delta_HO';
        % camada intermediaria
        % delta= phi_linha*epsilon
        % phi_linha = (1-v.^2) para tgh ou 1 para linear
        phi_linha = (1-v_int.^2);
        %for i=1:neuronios_ocultos
            soma=sum(pesos_oculta_saida'.*delta,2);
        %end;
        gamma = phi_linha .* soma;
        delta_IH = 2 * alr * par_atual_entrada' * gamma';
        pesos_entrada_oculta = pesos_entrada_oculta + delta_IH;

    end;
    % erros para gráfico
    err(iter,:) = mean(erro);
    err_teste(iter,:) = mean(erro_teste);
    err_valida(iter,:) = mean(erro_valida);
    % testes de parada
    gap=abs(err(iter,:)-err_valida(iter,:));
    disp(gap);
    disp(gap_anterior);
    disp(val_fail);
    if (gap > gap_anterior), 
        val_fail=val_fail+1; 
    else
        val_fail=0;
        gap_anterior=gap;
        p_o_s_iter_min=pesos_oculta_saida;
        p_e_o_iter_min=pesos_entrada_oculta;
        itermin=iter;
        errmin=err(iter,:);
    end;
    
    if (val_fail > 6), break; end;
    %Gráfico
    figure(1);
    semilogy([movmean(err,10) movmean(err_teste,10) movmean(err_valida,10)]);
    xlabel('Épocas');
    ylabel('Erro RMS');
    legend('Treinamento','Teste','Validação')
    grid;
    %xlim([1 etapas]);
    %ylim([.1 10]);
    %drawnow;    

    if reset
        pesos_entrada_oculta  = (randn(n_entradas,neuronios_ocultos) - 0.5)/10;
        pesos_oculta_saida = (randn(n_saidas,neuronios_ocultos) - 0.5)/10;
        fprintf('Pesos aleatórios na etapa %d\n',iter);
        reset = 0;
    end
    
    if earlystop
        fprintf('Parada na etapa %d\n',iter); 
        break 
    end 

    if err(iter) < 1e-6 % 0.001
        fprintf('convergiu após %d etapas\n',iter);
        break 
    end
    drawnow;      
end
m_erro_n(neuron,tenta)=err(5);
m_erro_teste_n(neuron,tenta)=err_teste(5);
m_erro_valida_n(neuron,tenta)=err_valida(5);
end
end;  