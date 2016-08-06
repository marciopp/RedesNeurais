%{
***************************************************************************
* Rede Neural por Backpropagation (BP) 
* Marcio Pinto Pereira - julho de 2016
* Programado em Matlab R2016a 
* Licenciado sob CC-BY-SA
***************************************************************************
%}
clear all; close all;
% Parâmetros
neuronios_ocultos = 30;
etapas = 10000;
% Entrada e Saída


%{
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


load('RedWhite1Wine.mat');
entrada1=[alcohol chlorides citricacid fixedacidity sulphates totalsulfurdioxide volatileacidity density residualsugar pH freesulfurdioxide];
saida1=quality;
%}

%{
x=[0:0.1:2*pi];
y1=  sin(x)+randn(1,length(x))/100;
y2= -sin(x)+randn(1,length(x))/100;

entrada1=x';
saida1=[ y2'];
%}

media_entrada = mean(entrada1);
desvio_padrao_entrada = std(entrada1);
for i=1:size(entrada1,2)
    entrada1(:,i) = (entrada1(:,i) - media_entrada(1,i)) / desvio_padrao_entrada(1,i);
end;
media_saida = mean(saida1);
desvio_padrao_saida = std(saida1);
for i=1:size(saida1,2)
    saida1(:,i) = (saida1(:,i) - media_saida(1,i)) / desvio_padrao_saida(1,i);
end;



p=randperm(length(entrada1));
px=p(1:round(length(entrada1)*0.6));
pteste=p(round(length(entrada1)*0.6):round(length(entrada1)*0.8));
pvalida=p(round(length(entrada1)*0.8):length(entrada1));



for i=1:length(px)
entrada(i,:)=entrada1(px(i),:);
saida(i,:)=saida1(px(i),:);
end;
for i=1:length(pteste)
teste(i,:)=entrada1(pteste(i),:);
saida_teste(i,:)=saida1(pteste(i),:);
end;
for i=1:length(pvalida)
valida(i,:)=entrada1(pvalida(i),:);
saida_valida(i,:)=saida1(pvalida(i),:);
end;
%{
entrada=entrada';
saida=saida';
teste=teste';
saida_teste=saida_teste';
valida=valida';
saida_valida=saida_valida';
%}



if size(entrada,1) ~= size(saida,1)
    disp('ERRO: pares entrada/saída com quantidades diferentes')
   return 
end

%{
% Normalizar pares entrada/saída
% Entradas
%entrada=entrada';
media_entrada = mean(entrada);
desvio_padrao_entrada = std(entrada);
for i=1:size(entrada,2)
    entrada(:,i) = (entrada(:,i) - media_entrada(1,i)) / desvio_padrao_entrada(1,i);
end;
%entrada=entrada';
media_teste = mean(teste);
desvio_padrao_teste = std(teste);
for i=1:size(teste,2)
    teste(:,i) = (teste(:,i) - media_teste(1,i)) / desvio_padrao_teste(1,i);
end;
media_valida = mean(valida);
desvio_padrao_valida = std(valida);
for i=1:size(valida,2)
    valida(:,i) = (valida(:,i) - media_valida(1,i)) / desvio_padrao_valida(1,i);
end;

% Saídas
%saida = saida';
media_saida = mean(saida);
desvio_padrao_saida = std(saida);
for i=1:size(saida,2)
    saida(:,i) = (saida(:,i) - media_saida(1,i)) / desvio_padrao_saida(1,i);
end;
%saida = saida';
media_saida_teste = mean(saida_teste);
desvio_padrao_saida_teste = std(saida_teste);
for i=1:size(saida_teste,2)
    saida_teste(:,i) = (saida_teste(:,i) - media_saida_teste(1,i)) / desvio_padrao_saida_teste(1,i);
end;
media_saida_valida = mean(saida_valida);
desvio_padrao_saida_valida = std(saida_valida);
for i=1:size(saida_valida,2)
    saida_valida(:,i) = (saida_valida(:,i) - media_saida_valida(1,i)) / desvio_padrao_saida_valida(1,i);
end;
%}

% Pares de treinamento com bias
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
%Inicializa pesos entre -0.05 e 0.05
pesos_entrada_oculta  = (randn(n_entradas, neuronios_ocultos) - 0.5)/10;
pesos_oculta_saida    = (randn(n_saidas,neuronios_ocultos)    - 0.5)/10;
erromin=1e99; errmin=1e99;
%%
%---------------
% Botão de parada
set(0, 'DefaultFigurePosition', [ 100 100 500 500 ]);
hstop = uicontrol('Style','PushButton','String','Pare', 'Position', [5 5 70 20],'callback','earlystop = 1;'); 
earlystop = 0;
%Botão para zerar pesos
hreset = uicontrol('Style','PushButton','String','Zerar Pesos', 'Position', get(hstop,'position')+[75 0 0 0],'callback','reset = 1;'); 
reset = 0;
% Botão de taxa de aprendizado
hlr = uicontrol('Style','slider','value',.1,'Min',.01,'Max',1,'SliderStep',[0.01 0.1],'Position', get(hreset,'position')+[75 0 100 0]);
%---------------

%%
for iter = 1:etapas
  disp(iter);
  for m=1:10
    %Taxa de aprendizagem
    alr = get(hlr,'value');
    %alr = 0.2;
    alr = 0.01;
    blr = alr / 10;
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
        % adicionar bias aqui !
        % camada saida linear
        u_ext = v_int'*pesos_oculta_saida'; % <--- saida !!!
        % saida
        % erro
        epsilon = par_atual_saida - u_ext; %ok
        
        % Teste
        % passo forward propagation
        % camada intermediaria tgh
        u_int_teste = par_atual_teste*pesos_entrada_oculta;
        v_int_teste = tanh(u_int_teste)'; %ok
        % adicionar bias aqui !
        % camada saida linear
        u_ext_teste = v_int_teste'*pesos_oculta_saida'; % <--- saida !!!
        % saida
        % erro
        epsilon_teste = par_atual_saida_teste - u_ext_teste;

        %Validação
        % passo forward propagation
        % camada intermediaria tgh
        u_int_valida = par_atual_valida*pesos_entrada_oculta;
        v_int_valida = tanh(u_int_valida)'; %ok
        % adicionar bias aqui !
        % camada saida linear
        u_ext_valida = v_int_valida'*pesos_oculta_saida'; % <--- saida !!!
        % saida
        % erro
        epsilon_valida = par_atual_saida_valida - u_ext_valida;
        
        
        %erro(:,j) = 0.5*epsilon.^2;
        erro(:,j) = sqrt(sum(epsilon.^2)); %erro rms
        erro_teste(:,j) = sqrt(sum(epsilon_teste.^2));
        erro_valida(:,j) = sqrt(sum(epsilon_valida.^2));
        
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
        %aux1=(par_atual'); 
        %aux2 = alr*pesos_oculta_saida'*epsilon'.*phi_linha;
        %delta_IH= alr*((1-(v_int.^2))*par_atual)*(delta*pesos_oculta_saida);
        %delta_IH= aux1*aux2'; % 51x3 - 51 x n * n x 3
        delta_IH = 2 * alr * par_atual_entrada' * gamma';
        pesos_entrada_oculta = pesos_entrada_oculta + delta_IH;
    end;
    errom(m)=mean(erro);
    errom_teste(m)=mean(erro_teste);
    errom_valida(m)=mean(erro_valida);
  end;  % gráficos
    %saida = pesos_oculta_saida*tanh(entrada*pesos_entrada_oculta)';
    %erro = par_atual_saida - u_ext;
    %err(iter) =  sum(((sum(delta.^2)).^0.5).^2).^0.5;
    err(iter,:) = mean(errom);
    err_teste(iter,:) = mean(errom_teste);
    err_valida(iter,:) = mean(errom_valida);

    if (err(iter,:) < errmin)
        p_o_s_iter_min=pesos_oculta_saida;
        p_e_o_iter_min=pesos_entrada_oculta;
        itermin=iter;
        errmin=err(iter,:);
    end;
    
    %err(iter,:) =  (sum(erro.^2,2))^0.5;
    figure(1);
    %clf;
    loglog([err err_teste err_valida]);
    xlabel('Épocas');
    ylabel('Erro RMS');
    grid;
    drawnow;    
    
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

    
    if err(iter) < 1e-3 % 0.001
        fprintf('convergiu após %d etapas\n',iter);
        break 
    end
       
end
%%
fprintf('Estado após %d etapas\n',iter);
a = (par_atual_saida* desvio_padrao_saida(:,1)) + media_saida(:,1);
b = (u_ext'* desvio_padrao_saida(:,1)) + media_saida(:,1);
par_atual_saida_saida_err = [a b b-a]
  