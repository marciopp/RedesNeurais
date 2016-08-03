%entrada=[pi/2];
x=[-4:0.1:10];
for i=1:length(x)
    entrada = x(i)
    entrada = (entrada-media_entrada)./desvio_padrao_entrada;
    entrada = [entrada 1];
    u_int = entrada*pesos_entrada_oculta;
    v_int = tanh(u_int)'; %ok
    % camada saida linear
    u_ext = v_int'*pesos_oculta_saida'; % <--- saida !!!
    saida(i,:) = (u_ext*desvio_padrao_saida')+media_saida;
    
end;
plot(saida)