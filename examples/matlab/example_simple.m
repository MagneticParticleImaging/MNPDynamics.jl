clear
close all

addpath('../../matlab/')

% Choose whether Brownian or Neel relaxation is simulated
MODE = 'brown';
%MODE = 'neel';


% CONSTANTS
%params.D_core = 20e-9;         % particle diameter in m
%params.alpha = 0.1;    % damping coefficient
%params.alpha = Inf;

%params.kAnis = 11000;  % anisotropy constant
%params.kAnis = 2100;
%params.kAnis = 625;
%params.kAnis = 0;

%params.viscosity = 1e-2;

t_length = 1000;% length of time vector
t_max = 2/25000;% maximum evaluation time in seconds

t = linspace(0,t_max,t_length);

% Anisotropy axis
params.n = [0;1;0];

f = 25000;

% Magnetic field for simulation
B =  @(t) 0.012*[0*t; sin(2*pi*f*t)+.1; sin(2*pi*f*t)];

tic;
[t,exp] = simulation(B, t, MODE, params);
time = toc;
disp(strcat('Solving ODE system took',{' '},num2str(time),' seconds.'))
xexp = exp(:,1);
yexp = exp(:,2);
zexp = exp(:,3);


figure
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
