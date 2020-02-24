function [t,exp,y] = simulation(B, t_vec, mode, varargin)
% Returns the solution of the Neel rotation model for given parameters.
%       INPUTS:
%       B: function of one scalar variable (time) that returns a 3D column
%           vector, describes the magnetic field over time
%       t_vec: vector of time points where the solution is to be evaluated
%       (optional): struct containing parameters; if any are not provided,
%       standard values are used. Possible parameters:
%           M_S: Core magnetization of a particle
%           D_core: Core diameter of a particle in m
%           Temp: Temperature in K
%           kAnis: Neel anisotropy constant
%           tau_N: scalar, Neel relaxation time constant, usually defined as
%              M_S*V_C/(2*alphha*gamma_tilde*k_B*T)
%           alpha: damping coefficient, usually 0.1, can be chosen as Inf for
%              faster computation and only small error
%           N: Number of spherical harmonics to be considered. More precisely,
%              this is the maximum index l to be considered for Y^l_m, resulting
%              in l(l+1) equations in the ODE system.
%           n: Neel easy axis, to be specified as a 3D column vector
%
%       OUTPUTS:
%       t: vector of time points where the solution was evaluated. Is equal
%          to input t_vec if the integration of the ODE succeeded.
%       exp: matrix of calculated mean magnetic moment.
%           Dimension: length(t) x 3.

if strcmp(mode,'neel')
    %% Neel
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
    if isfield(params, 'N')
        N = params.N;
    else
        N = 20;
    end
    if isfield(params,'n')
        n = params.n;
    else
        n = [0;0;1];
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
    if isfield(params, 'RelTol')
        RelTol = params.RelTol;
    else
        RelTol = 1e-3;
    end
    
    rot = rotz(n);  % Rotation matrix that rotates n to the z axis
    irot = inv(rot);% Rotation matrix that rotates the z axis to n
    
    B = @(t) rot*B(t);
    
    pr1 = M_S*V_core/(k_B*Temp);
    pr2 = kAnis*V_core/(k_B*Temp);
    
    
    counter = 0;
    nz = 0;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            
            nz = nz+1;
            if q~=-r
                nz = nz+1;
                if r~=0 && q~=r && q~=-r
                    nz = nz+1;
                end
            end
            if q~=r
                nz = nz+1;
            end
            if r<(N-1)
                nz = nz+1;
            end
            if r>1 && q>(-r+1)
                nz = nz+1;
            end
            if q>(-r+1)
                nz = nz+1;
            end
            if q<(r-1)
                nz = nz+1;
            end
            if r<N
                nz = nz+1;
                nz = nz+1;
                nz = nz+1;
            end
            
        end
    end
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            J(ind) = counter;
            I(ind) = counter;
            V(ind) = -1/(2*tau_N)*r*(r+1) + p4*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
            %V(ind) = pr2* 2*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
            ind = ind+1;
            if q~=-r
                if r~=0 && q~=r
                    J(ind) = counter-2*r;
                    I(ind) = counter;
                    V(ind) = -1i *p3* q*(r-q)/(2*r-1);%!
                    ind = ind+1;
                end
            end
            if r<(N-1)
                J(ind) = counter+2*(r+1)+2*(r+2);
                I(ind) = counter;
                V(ind) = -p4*r*(r+q+1)*(r+q+2)/((2*r+3)*(2*r+5));
                ind = ind+1;
            end
            if r>1 && q>(-r+1)
                J(ind) = counter-2*r-2*(r-1);
                I(ind) = counter;
                V(ind) = p4*(r+1)*(r-q)*(r-q-1)/((2*r-3)*(2*r-1));
                ind = ind+1;
            end
            if r<N
                J(ind) = counter+2*(r+1);
                I(ind) = counter;
                V(ind) = -1i*p3 * q*(r+q+1)/(2*r+3);%!
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    
    m_offset = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            J(ind) = counter;
            I(ind) = counter;
            V(ind) = -1i/(2) * p1 *2*q;%!
            ind = ind+1;
            if q~=-r
                if r~=0 && q~=r
                    J(ind) = counter-2*r;
                    I(ind) = counter;
                    V(ind) = p2*(r+1)*(r-q)/(2*r-1);
                    ind = ind+1;
                end
            end
            if r<N
                J(ind) = counter+2*(r+1);
                I(ind) = counter;
                V(ind) = -p2*r*(r+q+1)/(2*r+3);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_b3 = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            if q~=r
                J(ind) = counter+1;
                I(ind) = counter;
                V(ind) = -1i/(2) * p1 *(r-q)*(r+q+1);%!
                ind = ind+1;
            end
            if q<(r-1)
                J(ind) = counter-2*r+1;
                I(ind) = counter;
                V(ind) = p2 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
                ind = ind+1;
            end
            if r<N
                J(ind) = counter+2*(r+1)+1;
                I(ind) = counter;
                V(ind) = p2 * r*(r+q+1)*(r+q+2)/(4*r+6);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_bp = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            if q~=-r
                J(ind) = counter-1;
                I(ind) = counter;
                V(ind) = -1i/(2)*p1;%!
                ind = ind+1;
            end
            if q>(-r+1)
                J(ind) = counter-2*r-1;
                I(ind) = counter;
                V(ind) = -p2 * (r+1)/(4*r-2);
                ind = ind+1;
                
            end
            if r<N
                J(ind) = counter+2*(r+1)-1;
                I(ind) = counter;
                V(ind) = -p2 * r/(4*r+6);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_bm = sparse(I,J,V,(N+1)^2, (N+1)^2);
    tau = tau_N;
   
elseif strcmp(mode,'brown')
    %% Brown
    k_B = 1.38064852e-23;
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
    if isfield(params,'D_hydro')
        D_hydro = params.D_hydro;
    else
        D_hydro = D_core;
    end
    V_h = pi/6 * D_hydro^3;
    
    if isfield(params,'Temp')
        Temp = params.Temp;
    else
        Temp = 293;
    end
    if isfield(params, 'viscosity')
        eta = params.viscosity;
    else
        eta = 1e-3;
    end
    if isfield(params,'tau_B')
        tau_B = params.tau_B;
    else
        tau_B = 3*eta*V_h/(k_B*Temp);
    end
    if isfield(params, 'N')
        N = params.N;
    else
        N = 20;
    end
    if isfield(params, 'p2')
        p2 = params.p2;
    else
        p2 = M_S*V_core/(6*eta*V_h);
    end
    if isfield(params, 'RelTol')
        RelTol = params.RelTol;
    else
        RelTol = 1e-3;
    end


        
    rot = eye(3);
    irot = eye(3);
    
    pr1 = M_S*V_core/(k_B*Temp);    
    
    counter = 0;
    nz = 0;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            
            nz = nz+1;
            if q~=-r
                nz = nz+1;
                if r~=0 && q~=r && q~=-r
                    nz = nz+1;
                end
            end
            if q~=r
                nz = nz+1;
            end
            if r<(N-1)
                nz = nz+1;
            end
            if r>1 && q>(-r+1)
                nz = nz+1;
            end
            if q>(-r+1)
                nz = nz+1;
            end
            if q<(r-1)
                nz = nz+1;
            end
            if r<N
                nz = nz+1;
                nz = nz+1;
                nz = nz+1;
            end
            
        end
    end
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            J(ind) = counter;
            I(ind) = counter;
            V(ind) = -1/(2*tau_B)*r*(r+1);
            ind = ind+1;
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    
    m_offset = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter + 1;
            if q~=-r
                if r~=0 && q~=r
                    J(ind) = counter-2*r;
                    I(ind) = counter;
                    V(ind) = p2*(r+1)*(r-q)/(2*r-1);
                    ind = ind+1;
                end
            end
            if r<N
                J(ind) = counter+2*(r+1);
                I(ind) = counter;
                V(ind) = -p2*r*(r+q+1)/(2*r+3);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_b3 = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            if q<(r-1)
                J(ind) = counter-2*r+1;
                I(ind) = counter;
                V(ind) = p2 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
                ind = ind+1;
            end
            if r<N
                J(ind) = counter+2*(r+1)+1;
                I(ind) = counter;
                V(ind) = p2 * r*(r+q+1)*(r+q+2)/(4*r+6);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_bp = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    V = complex(zeros(nz,1));
    J = zeros(size(V));
    I = zeros(size(V));
    counter = 0;
    ind = 1;
    for r=0:N
        for q=-r:r
            counter = counter+1;
            if q>(-r+1)
                J(ind) = counter-2*r-1;
                I(ind) = counter;
                V(ind) = -p2 * (r+1)/(4*r-2);
                ind = ind+1;
                
            end
            if r<N
                J(ind) = counter+2*(r+1)-1;
                I(ind) = counter;
                V(ind) = -p2 * r/(4*r+6);
                ind = ind+1;
            end
        end
    end
    J = nonzeros(J);
    I = nonzeros(I);
    V = V(1:length(I));
    m_bm = sparse(I,J,V,(N+1)^2, (N+1)^2);
    
    tau = tau_B;
else
    error('unknown mode, use ''neel'' or ''brown''')
end

% initial value
y0 = zeros((N+1)^2,1);
y0(1) = 1/(4*pi);

% solve system
jac = @(t,y)odesys(t,y,1, B, m_offset, m_b3, m_bp, m_bm,tau);
opts = odeset("Vectorized", "on","Jacobian", jac, "RelTol", RelTol);
%opts = odeset( 'RelTol', 1e-3);
rhs = @(t,y)odesys(t,y,0, B, m_offset, m_b3, m_bp, m_bm,tau);
[t,y] = ode15s(rhs,t_vec , y0, opts);

% Calculate expectation from spherical harmonics
xexptemp = real((4*pi/3)*(.5*y(:,2)-y(:,4)));
yexptemp = real(-1i*(4*pi/3)*(y(:,4)+.5*y(:,2)));
zexptemp = real((4*pi/3)*y(:,3));

% Rotate the coordinate system back (solver only works for the z-axis as
% the easy axis)
xexp = irot(1,1)*xexptemp + irot(1,2)*yexptemp + irot(1,3)*zexptemp;
yexp = irot(2,1)*xexptemp + irot(2,2)*yexptemp + irot(2,3)*zexptemp;
zexp = irot(3,1)*xexptemp + irot(3,2)*yexptemp + irot(3,3)*zexptemp;

exp = [xexp, yexp, zexp];
end
