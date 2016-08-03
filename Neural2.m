% -------------------------
% Padrões de Entrada
% -------------------------
%{
u=[0 0;
   0 1;
   1 0;
   1 1]';
yd=[0 1 1 0];
%}

x=[0:0.1:2*pi];
y1=  sin(x)+randn(1,length(x))/100;
y2= -sin(x)+randn(1,length(x))/100;
u=x;
yd=[y1];

ub=[ones(1,max(size(u))); u]; 				% Inclusão do 'bias'
Np=max(size(u(1,:))); 							% Número de padrões
if max(size(u(1,:)))~=max(size(yd(1,:))),
   error('Erro nos padrões de entrada!');
end
% -------------------------
% Estrutura da Rede
% -------------------------
clc
Ne1=input('Neurônios na camada escondida 1 (Ne1)= ');
if Ne1<=0,
   error('Número de Neurônios Inválido!');
end
Ne2=input('Neurônios na camada escondida 1 (Ne2)= ');
if Ne2<=0,
   error('Número de Neurônios Inválido!');
end
Nh=Ne1+1;								% Inclusão do 'bias'
Nm=Ne2+1;								% Inclusão do 'bias'
Nu=max(size(u(:,1)))+1;			% Número de neurônios na camada de entrada + 'bias'
Ny=max(size(yd(:,1)));			% Número de neurônios na camada de saída
% -------------------------
% Fator de Momento
% -------------------------
alfa=input('Fator de momento (alfa)= ');
if (alfa<0)|(alfa>1),
   error('Fator de momento inválido');
   return;
end
deltaWuh=zeros(Nh,Nu);
deltaWhm=zeros(Nm,Nh);
deltaWmy=zeros(Ny,Nm);
deltaWuh_ant=zeros(Nh,Nu);
deltaWhm_ant=zeros(Nm,Nh);
deltaWmy_ant=zeros(Ny,Nm);
% -------------------------
% Inicialização dos Pesos
% -------------------------
op=menu('Pesos Iniciais','Aleatórios','Arquivo','Workspace','Sair do Programa'); 
if op==1, 
   uh=randn(Nh,Nu);					% uh = Pesos Entrada -> Camada Escondida1
   hm=randn(Nm,Nh);					% hm = Pesos Camada Escondida 1 -> Camada Escondida 2
   my=randn(Ny,Nm);					% my = Pesos Camada Escondida 2 -> Saída
elseif op==2,
   narq=input('Digite o nome do arquivo de dados (.mat) entre aspas simples: ');
   load(narq);
elseif op==4,
    return;
end
ask=0;
ask=input('Digite 1 para salvar pesos iniciais: ');
if ask==1,
   uho=uh;
   hmo=hm;
   myo=my;
end
% -------------------------
% Algoritmo de Treinamento
% -------------------------
n=1; 									% Taxa de Aprendizado
clear err
errmin=100;
Nep=input('Número de Épocas (Nep)= ');
if Nep<=0,
   error('Número de Épocas Inválido!');
end
for k=1:Nep,
   for i=1:Np,
      
      auxh=uh*ub(:,i);
      auxh(1,:)=1; 				% Ativa o 'bias' da camada escondida
      h=1./(1+exp(-auxh));		% Função de ativação da camada escondida 1 --> Sigmóide
      
      auxm=hm*h;
      auxm(1,:)=1; 				% Ativa o 'bias' da camada escondida
      m=1./(1+exp(-auxm));		% Função de ativação da camada escondida 2 --> Sigmóide
      
      auxy=my*m;
      y=auxy;						% Função de ativação da camada de saída --> Linear
      
      e=yd(:,i)-y;				% Erro
      E(:,i)=0.5*e.^2;  		% Erro Médio Quadrático
      
      deltamy=e*1;				% deltamy = (yd-y)*f'(net_my)
      
      for p=1:Ny,					% Cálculo da variação dos Pesos Camada Escondida 2 -> Saída
         for q=1:Nm,
            deltaWmy(p,q)=n*deltamy(p,:)*m(q,:)+alfa*deltaWmy_ant(p,q);
         end
      end
      
      aux2=m.*(1-m);				% f'(net_hm)
      aux3=(deltamy'*my)'; 	
      deltahm=aux2.*aux3;		% Cálculo de deltahm
     
    
      for p=1:Nm,					% Cálculo da variação dos Pesos Camada Escondida 1 -> Camada Escondida 2
         for q=1:Nh,
            deltaWhm(p,q)=n*deltahm(p,:)*h(q,:)+alfa*deltaWhm_ant(p,q);
         end
      end
      
      aux2=h.*(1-h);				% f'(net_uh)
      aux3=(deltahm'*hm)'; 	
      deltauh=aux2.*aux3;		% Cálculo de deltauh
   
      for p=1:Nh,					% Cálculo da variação dos Pesos Entrada -> Camada Escondida
         for q=1:Nu,
            deltaWuh(p,q)=n*deltauh(p,:)*ub(q,i)+alfa*deltaWuh_ant(p,q);
         end
      end
      
      my=my+deltaWmy;      	% Atualização dos Pesos Camada Escondida 2 -> Saída
      hm=hm+deltaWhm;      	% Atualização dos Pesos Camada Escondida 1 -> Camada Escondida 2
      uh=uh+deltaWuh;			% Atualização dos Pesos Entrada -> Camada Escondida 1
      
   end
   
   err(k,:)=mean(E',1);			% Média do erro quadrático por época de treinamento
   
   if k>1,				
      if err(k,:)<errmin, % Guarda os pesos que fornecem menor erro
         pos=k;       
         errmin=err(k,:);
         uhemin=uh;
         hmemin=hm;
         myemin=my;
      end
   end
  
end
% -------------------------
% Mostra Resultados
% -------------------------
clf;
semilogy(err);
xlabel('Número de Épocas');
ylabel('Erro médio quadrático');
grid;
%testep3;