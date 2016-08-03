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
x=[0:0.1:2*pi];
y1=  sin(x)+randn(1,length(x))/100;
y2= -sin(x)+randn(1,length(x))/100;
entrada=x';
saida=[y1' y2'];
%}
%{
entrada=[0 0;
   0 1;
   1 0;
   1 1];
saida=[0 1 1 0]';
%}
load('RedWine.mat');
entrada=[alcohol chlorides citricacid density fixedacidity freesulfurdioxide pH residualsugar sulphates totalsulfurdioxide volatileacidity];
saida=quality;
if size(entrada,1) ~= size(saida,1)
    disp('ERRO: pares entrada/saída com quantidades diferentes')
   return 
end    
% Normalizar pares entrada/saída
% Entradas
%entrada=entrada';
media_entrada = mean(entrada);
desvio_padrao_entrada = std(entrada);
for i=1:size(entrada,2)
    entrada(:,i) = (entrada(:,i) - media_entrada(1,i)) / desvio_padrao_entrada(1,i);
end;
%entrada=entrada';
% Saídas
%saida = saida';
media_saida = mean(saida);
desvio_padrao_saida = std(saida);
for i=1:size(saida,2)
    saida(:,i) = (saida(:,i) - media_saida(1,i)) / desvio_padrao_saida(1,i);
end;
%saida = saida';
% Pares de treinamento com bias
pares_treinamento = size(entrada,1);
bias = ones(pares_treinamento,1);
entrada = [entrada bias];
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
        %erro(:,j) = 0.5*epsilon.^2;
        erro(:,j) = sqrt(sum(epsilon.^2));
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
    % gráficos
    %saida = pesos_oculta_saida*tanh(entrada*pesos_entrada_oculta)';
    %erro = par_atual_saida - u_ext;
    %err(iter) =  sum(((sum(delta.^2)).^0.5).^2).^0.5;
    err(iter,:) = mean(erro);
    if (err(iter,:) < errmin)
        p_o_s_iter_min=pesos_oculta_saida;
        p_e_o_iter_min=pesos_entrada_oculta;
        itermin=iter;
        errmin=err(iter,:);
    end;
    
    %err(iter,:) =  (sum(erro.^2,2))^0.5;
    figure(1);
    %clf;
    semilogy(err);
    xlabel('Número de Épocas');
    ylabel('Erro médio quadrático');
    grid;
        
    
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

    
    if err(iter) < 1e-18 % 0.001
        fprintf('converged at epoch: %d\n',iter);
        break 
    end
       
end
%%
fprintf('state after %d etapas\n',iter);
a = (par_atual_saida* desvio_padrao_saida(:,1)) + media_saida(:,1);
b = (u_ext'* desvio_padrao_saida(:,1)) + media_saida(:,1);
par_atual_saida_saida_err = [a b b-a]
  
