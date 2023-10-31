function  [obj_save,U,V,time_save,residual,objADMM] = iADMMn(Y,c,lambda_d,lambda_t,options)
  % iADMMn for solving 
  % (*)  \min_{U,V} \sum_{i=1}^m \sum_{j=1}^n (1+c y_{ij}-y_{ij}) \log (1+\exp(u_i v_j^\top))
  %                        -c y_{ij}u_i v_j^\top + \frac{\lambda_d}{2} \|U\|_F^2 
  %                        + \frac{\lambda_t}{2}\|V\|_F^2
  % note that y_{ij} \in [0,1]
  %
  % written by LTK. Hien
  % lastest version Oct 2023
  %
  %
  % input: 0<=Y<=1, c, lambda_d, lambda_t
  %        options.beta : penalty parameter of iADMMn
  %        options.max_time : max time of running the algorithm
  %        options.max_iter : max number of iterations to run the algorithm
  %        options.tau1, options.tau2: the parameter tau_1, tau_2 of iADMMn to update the
  %                                     multiplier
  %        options.U0, options.V0: initial points of iADMMn 
  %
  % output: obj_save: sequence of the objective of problem (*),
  %            U,V : solution
  %         time_save: sequence of running time
  %         residual: sequence of relative residuals ||W-UV||/||W|| of iADMMn
  %          objADMM: objective of the reformulated problem 
  %
   cputime0 = tic; 
   [m,n] = size(Y); 
   max_time=options.max_time;
   max_iter=options.max_iter;

   %parameter
   beta=options.beta;
   tau1=options.tau1;
   tau2=options.tau2;

   % initial point
   U=options.U0;
   U_prev=U;
   V=options.V0;
   V_prev=V;
   w=zeros(m,n);
   W=U*V;
   residual=inf;

   %start
   i=1;
   time_i=toc(cputime0);
   time_save=time_i;
   UV=U*V;
   yy=1+(c-1)*Y;

   GW=yy.*logexp(UV);
   YUV=Y.*UV;
   obj=sum(GW(:))-c*sum(YUV(:))+lambda_d*norm(U,'fro')^2/2+lambda_t/2*norm(V,'fro')^2;
   obj_save=obj;

   objADMM=obj;

   t1=1;
   Lu = norm(V*V'); 
   Lv=norm(U'*U);
 
   LG=1/4*max(yy(:));
   C2=(tau1+1)*tau2/tau1/(2*beta*(1-abs(tau1-tau2))*(1-abs(1-tau2/tau1)));
   C3=LG+beta;
   %check parameters conditions 
   if 8*C2*LG*LG>=C3 
       error("parameters do not satisfy convergence condition: 8*C2*LG*LG < C3");
       
   end

   while i<=max_iter &&  time_i<=max_time
     t2=1/2*(1+sqrt(1+4*t1*t1));
     ex_coef1=(t1-1)/t2;
     t1=t2;
     % update U
     Vt=V';
     VVt=V*Vt;
     Lu_new= norm(VVt); 
     if options.inertial==0
         ex_coef=0;
     else
      ex_coef=min(ex_coef1,0.9999*sqrt(Lu/Lu_new));
     end

     Lu=Lu_new;
     beta_Lv=beta*Lu;
     U_ex=U+ex_coef*(U-U_prev);
     U_prev=U;
     U=1/(beta_Lv+lambda_d)*(beta_Lv*U_ex- w*Vt-beta*(U_ex*VVt-W*Vt));
     
     %update V
     Ut=U'; 
     UtU=Ut*U;
     Lv_new=norm(UtU);
     if options.inertial==0
         ex_coef=0;
     else
        ex_coef=min(ex_coef1,0.9999*sqrt(Lv/Lv_new));
     end
     Lv=Lv_new;
     beta_Lv=beta*Lv; 
     V_ex=V+ ex_coef*(V-V_prev);
     V_prev=V;
     V=1/(beta_Lv+lambda_t)*(beta_Lv*V_ex- Ut*w-beta*(UtU*V_ex-Ut*W));

     expW=expX(W);
     nablaG=yy.*expW-c*Y;
     UV=U*V;
    
     
     W=1/(beta+LG)*(LG*W - nablaG +w+ beta*UV);
     diff=UV-W;
     %update w
     w=tau1*w+tau2*beta*diff;
     % compute objective
     GW=yy.*logexp(UV);
     YUV=Y.*UV;
     obj=sum(GW(:))-c*sum(YUV(:))+lambda_d*norm(U,'fro')^2/2+lambda_t/2*norm(V,'fro')^2;
     obj_save=[obj_save,obj];

     % compute ADMM objective
     GW=yy.*logexp(W);
     YW=Y.*W;
     obj_2=sum(GW(:))-c*sum(YW(:))+lambda_d*norm(U,'fro')^2/2+lambda_t/2*norm(V,'fro')^2;
     objADMM=[objADMM,obj_2];

     res=norm(W-U*V,'fro')/norm(W,'fro');
     residual=[residual,res];
     time_i=toc(cputime0);
     time_save=[time_save,time_i];
     if mod(i,2)==0
            fprintf('iADMMn: i %4d, fitting error: %1.2e rel residual %.2e, objective %.2e \n',i,obj,res,obj_2);     
     end
     i=i+1;
   end 
end

function e=expX(X)
%compute e^x/(1+e^x)
 e=exp(-max(-X,0))./(1+exp(-abs(X)));
end

function e=logexp(X)
%compute log(1+e^x)
 e=log(1+exp(-abs(X)))+max(0,X);
end
