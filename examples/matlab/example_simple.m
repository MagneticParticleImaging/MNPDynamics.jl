clear
%close all

addpath('../../matlab')

% Choose whether Brownian or Neel relaxation is simulated
%MODE = 'brown';
MODE = 'neel';
%MODE = 'brown';


% CONSTANTS
%params.D_core = 20e-9;         % particle diameter in m
%params.alpha = 0.1;    % damping coefficient
%params.alpha = Inf;

%params.kAnis = 11000;  % anisotropy constant
%params.kAnis = 2100;
%params.kAnis = 625;
%params.kAnis = 0;
params.kAnis = 11000;
params.RelTol = 1e-3;
%params.alpha = 0.1;
%params.p1 = 0;
%params.p3 = 0;
%params.alpha = Inf;
%params.tau_N = 1.4163e-08;
%params.tau_N = 1e-4;
params.N=20;
%params.viscosity = 1e-2;

t_length = 1000;% length of time vector
t_max = 4/25000;% maximum evaluation time in seconds

t = linspace(0,t_max,t_length);

% Anisotropy axis
%n_func = @(t)[cos(.25*2*pi*t'/t_max);sin(.25*2*pi*t'/t_max);0*t'].';
%n_func = @(t) [ones(size(t));0*t;0*t];
%params.n = @(t) n_func(t)/norm(n_func(t));
%params.n = @(t) n_func(t);
%params.n =[1;0;1]/sqrt(2);
params.n = [1;0;0];
%params.n = [0;0;1];
%params.kAnis = 0;

f = 25000;

% Magnetic field for simulation
%B =  @(t) 0.012*[.6*sin(2*pi*f*t); sin(2*pi*f*t)+.1; sin(2*pi*f*t)];
B = @(t) 0.012*[sin(2*pi*f*t); 0*t;0*t];
%B = @(t) .5*0.012*[0*t;t<1/50000; 0*t];

%B = @(t) .012*[0;(t<(t_max/t_length*500))*0.01;sin(2*pi*f*t).*(t<(t_max/t_length*500))];
%B = @(t) 0.012*[sin(2*pi*f*t);0;0];

tic;
[t,exp,y] = simulation(B, t, MODE, params);
time = toc;
disp(strcat('Solving ODE system took',{' '},num2str(time),' seconds.'))
xexp = exp(:,1);
yexp = exp(:,2);
zexp = exp(:,3);


%figure
subplot(1,2,1)
plot(t, [xexp,yexp,zexp])
legend({'$\bar{m}_x$','$\bar{m}_y$','$\bar{m}_z$'}, 'Interpreter','latex')
subplot(1,2,2)
dt = diff(t);
dxdt = diff(xexp)./dt;
dydt = diff(yexp)./dt;
dzdt = diff(zexp)./dt;
plot(t(1:end-1),[dxdt, dydt, dzdt]) 
legend({'$\frac{\partial \bar{m}_x}{\partial t}$','$\frac{\partial \bar{m}_y}{\partial t}$','$\frac{\partial \bar{m}_z}{\partial t}$'}, 'Interpreter','latex')
% Note: the result has to be multiplied by M_S*V_C to obtain the magnetic
% moment
