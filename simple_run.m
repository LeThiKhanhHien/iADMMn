clear all; close all; clc; 
%data
m =200;
n = 200;
r=100;
c=1;
lambda_d=1/(2^3);
lambda_t=lambda_d;
Y = sprand(m,n,0.1);
Y(Y>0)=1;  

% parameter
options.max_time=3;
options.max_iter=inf;
options.beta=1;

options.tau1=0.1;
options.tau2=0.1;

% initial point
U0=rand(m,r);
V0 = rand(r,n);
% scale initial point 
 options.U0=U0/norm(U0);
 options.V0=V0/norm(V0);

 % run iADMMn with inertial term 
 options.inertial=1;
[obj,U,V,time_save,residual,ADMMobj] = iADMMn(Y,c,lambda_d,lambda_t,options);

%run non-inertial version 
options.inertial=0;
[obj2,U2,V2,time_save2,residual2,ADMMobj2] = iADMMn(Y,c,lambda_d,lambda_t,options);

%run GD 
[obj_GD,U_GD,V_GD,time_save_GD] = GD(Y,c,lambda_d,lambda_t,options);

%draw some image
figure;
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
semilogy(time_save,log(obj),'r','LineWidth',2);hold on; 
semilogy(time_save2,log(obj2),'b','LineWidth',2);hold on; 
semilogy(time_save_GD,log(obj_GD),'k','LineWidth',2);hold on; 
ylabel('log of the objective');
xlabel('Time')
legend('iADMMn (0.1,0.1)','ADMM (0.1,0.1)', 'GD');


