function  [obj_save,U,V,time_save] = GD(Y,c,lambda_d,lambda_t,options)
  % GD (alternating gradient descent) algorithm for solving 
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
  %        options.U0, options.V0: initial points of iADMMn 
  %
  % output: obj_save: sequence of the objective of problem (*),
  %            U,V : solution
  %         time_save: sequence of running time
  %
   cputime0 = tic; 
   max_time=options.max_time;
   max_iter=options.max_iter;

   % initial point
   U=options.U0;
   V=options.V0;
    %start
   i=1;
   time_i=toc(cputime0);
   time_save=time_i;
   UV=U*V;
   yy=1+(c-1)*Y;
   LG=1/4*max(yy(:));

   GW=yy.*logexp(UV);
   YUV=Y.*UV;
   obj=sum(GW(:))-c*sum(YUV(:))+lambda_d*norm(U,'fro')^2/2+lambda_t/2*norm(V,'fro')^2;
   obj_save=obj;
   while i<=max_iter &&  time_i<=max_time
       % update U
       W=U*V;
       P=expX(W);
   
       gradU=P*V'+(c-1)*(Y.*P)*V'-c*Y*V'+lambda_d*U;
       stepsize=1/(LG*norm(V)^2+lambda_d);
       U=U-stepsize*gradU;

       %update V
       P=expX(U*V);
       gradV=U'*P+(c-1)*U'*((Y).*(P))-c*U'*Y+lambda_t*V;
       stepsize=1/(LG*norm(U)^2+lambda_t);
       V=V-stepsize*gradV;

       UV=U*V;
       yy=1+(c-1)*Y;

       GW=yy.*logexp(UV);
       YUV=Y.*UV;
       obj=sum(GW(:))-c*sum(YUV(:))+lambda_d*norm(U,'fro')^2/2+lambda_t/2*norm(V,'fro')^2;

       obj_save=[obj_save,obj];
       time_i=toc(cputime0);
       time_save=[time_save,time_i];
         if mod(i,2)==0
                fprintf('GD: iteration %4d fitting error: %1.2e \n',i,obj);     
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
