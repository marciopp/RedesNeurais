% -------------------------
% Padr�es de Entrada
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

ub=[ones(1,max(size(u))); u]; 				% Inclus�o do 'bias'
Np=max(size(u(1,:))); 							% N�mero de padr�es
if max(size(u(1,:)))~=max(size(yd(1,:))),
   error('Erro nos padr�es de entrada!');
end
% -------------------------
% Estrutura da Rede
% -------------------------
clc
Ne1=input('Neur�nios na camada escondida 1 (Ne1)= ');
if Ne1<=0,
   error('N�mero de Neur�nios Inv�lido!');
end
Ne2=input('Neur�nios na camada escondida 1 (Ne2)= ');
if Ne2<=0,
   error('N�mero de Neur�nios Inv�lido!');
end
Nh=Ne1+1;								% Inclus�o do 'bias'
Nm=Ne2+1;								% Inclus�o do 'bias'
Nu=max(size(u(:,1)))+1;			% N�mero de neur�nios na camada de entrada + 'bias'
Ny=max(size(yd(:,1)));			% N�mero de neur�nios na camada de sa�da
% -------------------------
% Fator de Momento
% -------------------------
alfa=input('Fator de momento (alfa)= ');
if (alfa<0)|(alfa>1),
   error('Fator de momento inv�lido');
   return;
end
deltaWuh=zeros(Nh,Nu);
deltaWhm=zeros(Nm,Nh);
deltaWmy=zeros(Ny,Nm);
deltaWuh_ant=zeros(Nh,Nu);
deltaWhm_ant=zeros(Nm,Nh);
deltaWmy_ant=zeros(Ny,Nm);
% -------------------------
% Inicializa��o dos Pesos
% -------------------------
op=menu('Pesos Iniciais','Aleat�rios','Arquivo','Workspace','Sair do Programa'); 
if op==1, 
   uh=randn(Nh,Nu);					% uh = Pesos Entrada -> Camada Escondida1
   hm=randn(Nm,Nh);					% hm = Pesos Camada Escondida 1 -> Camada Escondida 2
   my=randn(Ny,Nm);					% my = Pesos Camada Escondida 2 -> Sa�da
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
Nep=input('N�mero de �pocas (Nep)= ');
if Nep<=0,
   error('N�mero de �pocas Inv�lido!');
end
for k=1:Nep,
   for i=1:Np,
      
      auxh=uh*ub(:,i);
      auxh(1,:)=1; 				% Ativa o 'bias' da camada escondida
      h=1./(1+exp(-auxh));		% Fun��o de ativa��o da camada escondida 1 --> Sigm�ide
      
      auxm=hm*h;
      auxm(1,:)=1; 				% Ativa o 'bias' da camada escondida
      m=1./(1+exp(-auxm));		% Fun��o de ativa��o da camada escondida 2 --> Sigm�ide
      
      auxy=my*m;
      y=auxy;						% Fun��o de ativa��o da camada de sa�da --> Linear
      
      e=yd(:,i)-y;				% Erro
      E(:,i)=0.5*e.^2;  		% Erro M�dio Quadr�tico
      
      deltamy=e*1;				% deltamy = (yd-y)*f'(net_my)
      
      for p=1:Ny,					% C�lculo da varia��o dos Pesos Camada Escondida 2 -> Sa�da
         for q=1:Nm,
            deltaWmy(p,q)=n*deltamy(p,:)*m(q,:)+alfa*deltaWmy_ant(p,q);
         end
      end
      
      aux2=m.*(1-m);				% f'(net_hm)
      aux3=(deltamy'*my)'; 	
      deltahm=aux2.*aux3;		% C�lculo de deltahm
     
    
      for p=1:Nm,					% C�lculo da varia��o dos Pesos Camada Escondida 1 -> Camada Escondida 2
         for q=1:Nh,
            deltaWhm(p,q)=n*deltahm(p,:)*h(q,:)+alfa*deltaWhm_ant(p,q);
         end
      end
      
      aux2=h.*(1-h);				% f'(net_uh)
      aux3=(deltahm'*hm)'; 	
      deltauh=aux2.*aux3;		% C�lculo de deltauh
   
      for p=1:Nh,					% C�lculo da varia��o dos Pesos Entrada -> Camada Escondida
         for q=1:Nu,
            deltaWuh(p,q)=n*deltauh(p,:)*ub(q,i)+alfa*deltaWuh_ant(p,q);
         end
      end
      
      my=my+deltaWmy;      	% Atualiza��o dos Pesos Camada Escondida 2 -> Sa�da
      hm=hm+deltaWhm;      	% Atualiza��o dos Pesos Camada Escondida 1 -> Camada Escondida 2
      uh=uh+deltaWuh;			% Atualiza��o dos Pesos Entrada -> Camada Escondida 1
      
   end
   
   err(k,:)=mean(E',1);			% M�dia do erro quadr�tico por �poca de treinamento
   
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
xlabel('N�mero de �pocas');
ylabel('Erro m�dio quadr�tico');
grid;
%testep3;