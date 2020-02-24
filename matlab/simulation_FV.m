function [t, exp, y] = simulation_FV(B, t_vec, tr, varargin)

k_B = 1.38064852e-23;
gam_gyro=1.75*10^11;
if nargin > 3
    params = varargin{1};
else
    params.i = {};
end

if isfield(params,'M_S')
    M_S = params.M_S;
else
    M_S = 474000;
end
if isfield(params, 'D_core')
    D_core = params.D_core;
else
    D_core = 20e-9;
end
V_core = pi/6 * D_core^3;
if isfield(params,'Temp')
    Temp = params.Temp;
else
    Temp = 293;
end
if isfield(params, 'alpha')
    alpha = params.alpha;
else
    alpha = 0.1;
end
if isfield(params, 'kAnis')
    kAnis = params.kAnis;
else
    kAnis = 625;
end
if isfield(params,'tau_N')
    tau_N = params.tau_N;
else
    tau_N = M_S*V_core/(k_B*Temp*gam_gyro)*(1+alpha^2)/(2*alpha);
end
if isfield(params,'n')
    n = params.n;
else
    n = [0;0;1];
end
if isfield(params, 'RelTol')
    RelTol = params.RelTol;
else
    RelTol = 1e-3;
end
if isfield(params, 'beta')
    beta = params.beta;
else
    beta = 0;
end
if isfield(params, 'p1')
    p1 = params.p1;
else
    p1 = gam_gyro/(1+alpha^2);
end
if isfield(params,'p2')
    p2 = params.p2;
else
    p2 = alpha*gam_gyro/(1+alpha^2);
end
if isfield(params, 'p3')
    p3 = params.p3;
else
    p3 = 2*gam_gyro/(1+alpha^2)*kAnis/M_S;
end
if isfield(params, 'p4')
    p4 = params.p4;
else
    p4 = 2*alpha*gam_gyro/(1+alpha^2)*kAnis/M_S;
end

N = size(tr.fMat,1);
tr.C = 1/(2*tau_N)*tr.areasi.*tr.C;
y0 = 1/(4*pi)*ones(1,N);
rhs = @(t,y) FV_matrix(t, B, N, p1, p2, p3, p4, beta, tr.mids, tr.ds, n, tr.iis, tr.valcs, tr.C, tr.C, tr.a_ijs, tr.e_is, tr.areasidil, tr.tr2edge, tr.flow_signs)*y;
jac = @(t,y) FV_matrix(t, B, N, p1, p2, p3, p4, beta, tr.mids, tr.ds, n, tr.iis, tr.valcs, tr.C, tr.C, tr.a_ijs, tr.e_is, tr.areasidil, tr.tr2edge, tr.flow_signs);
opts = odeset('Jacobian', jac, 'RelTol', RelTol);
[t,y] = ode15s(rhs, t_vec, y0, opts);

exp = zeros(length(t), 3);
for i=1:N
    exp = exp + y(:,i).*tr.areas(i).*tr.centers(i,:);
end


end
