function [t,y] = return_neel(Bb,pr1,pr2,tau_N,alpha, t_vec, N)
% Returns the solution of the Neel rotation model for given parameters.
%       INPUTS:
%       Bb: function of one scalar variable (time) that returns a 3D column
%           vector, describes the magnetic field over time
%       pr1: scalar, M_S*V_C/(k_B*T)
%       pr2: scalar, K_anis*V_C/(k_B*T)
%       tau_N: scalar, Neel relaxation time constant,
%              M_S*V_C/(2*alphha*gamma_tilde*k_B*T)
%       alpha: damping coefficient, usually 0.1, can be chosen as Inf for
%              faster computation and only small error
%       t_vec: vector of time points where the solution is to be evaluated
%       N: Number of spherical harmonics to be considered. More precisely,
%          this is the maximum index l to be considered for Y^l_m, resulting
%          in l(l+1) equations in the ODE system.
%
%       OUTPUTS:
%       t: vector of time points where the solution was evaluated. Is equal
%          to input t_vec if the integration of the ODE succeeded.
%       y: matrix of spherical harmonic expansions over time. One line
%           corresponds to one time instance, one column corresponds to one
%           spherical harmonic coefficient over time.
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
I = zeros(size(V));
J = zeros(size(V));
counter = 0;
ind = 1;
for r=0:N
    for q=-r:r
        counter = counter+1;
        I(ind) = counter;
        J(ind) = counter;
        V(ind) = -r*(r+1) + pr2* 2*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
        ind = ind+1;
        if q~=-r
            if r~=0 && q~=r
                I(ind) = counter-2*r;
                J(ind) = counter;
                V(ind) = 2*1i/alpha * pr2 * q*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<(N-1)
            I(ind) = counter+2*(r+1)+2*(r+2);
            J(ind) = counter;
            V(ind) = -pr2 * 2*r*(r+q+1)*(r+q+2)/((2*r+3)*(2*r+5));
            ind = ind+1;
        end
        if r>1 && q>(-r+1)
            I(ind) = counter-2*r-2*(r-1);
            J(ind) = counter;
            V(ind) = pr2 * 2*(r+1)*(r-q)*(r-q-1)/((2*r-3)*(2*r-1));
            ind = ind+1;
        end
        if r<N
            I(ind) = counter+2*(r+1);
            J(ind) = counter;
            V(ind) = 2*1i/alpha * pr2 * q*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
end
I = nonzeros(I);
J = nonzeros(J);
V = V(1:length(J));

m_offset = sparse(J,I,V,(N+1)^2, (N+1)^2);

V = complex(zeros(nz,1));
I = zeros(size(V));
J = zeros(size(V));
counter = 0;
ind = 1;
for r=0:N
    for q=-r:r
        counter = counter+1;
        I(ind) = counter;
        J(ind) = counter;
        V(ind) = 1i/(2*alpha) * pr1 *2*q;
        ind = ind+1;
        if q~=-r
            if r~=0 && q~=r
                I(ind) = counter-2*r;
                J(ind) = counter;
                V(ind) = pr1*(r+1)*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<N
            I(ind) = counter+2*(r+1);
            J(ind) = counter;
            V(ind) = -pr1*r*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
end
I = nonzeros(I);
J = nonzeros(J);
V = V(1:length(J));
m_b3 = sparse(J,I,V,(N+1)^2, (N+1)^2);

V = complex(zeros(nz,1));
I = zeros(size(V));
J = zeros(size(V));
counter = 0;
ind = 1;
for r=0:N
    for q=-r:r
        counter = counter+1;
        if q~=r
            I(ind) = counter+1;
            J(ind) = counter;
            V(ind) = 1i/(2*alpha) * pr1 *(r-q)*(r+q+1);
            ind = ind+1;
        end
        if q<(r-1)
            I(ind) = counter-2*r+1;
            J(ind) = counter;
            V(ind) = pr1 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
            ind = ind+1;
        end
        if r<N
            I(ind) = counter+2*(r+1)+1;
            J(ind) = counter;
            V(ind) = pr1 * r*(r+q+1)*(r+q+2)/(4*r+6);
            ind = ind+1;
        end
    end
end
I = nonzeros(I);
J = nonzeros(J);
V = V(1:length(J));
m_bp = sparse(J,I,V,(N+1)^2, (N+1)^2);

V = complex(zeros(nz,1));
I = zeros(size(V));
J = zeros(size(V));
counter = 0;
ind = 1;
for r=0:N
    for q=-r:r
        counter = counter+1;
        if q~=-r
            I(ind) = counter-1;
            J(ind) = counter;
            V(ind) = 1i/(2*alpha)*pr1;
            ind = ind+1;
        end
        if q>(-r+1)
            I(ind) = counter-2*r-1;
            J(ind) = counter;
            V(ind) = -pr1 * (r+1)/(4*r-2);
            ind = ind+1;

        end
        if r<N
            I(ind) = counter+2*(r+1)-1;
            J(ind) = counter;
            V(ind) = -pr1 * r/(4*r+6);
            ind = ind+1;
        end
    end
end
I = nonzeros(I);
J = nonzeros(J);
V = V(1:length(J));
m_bm = sparse(J,I,V,(N+1)^2, (N+1)^2);

% initial value
y0 = zeros((N+1)^2,1);
y0(1) = 1/(4*pi);

% solve system
jac = @(t,y)neel_odesys(t,y,1, Bb, m_offset, m_b3, m_bp, m_bm,tau_N);
opts = odeset("Vectorized", "on","Jacobian", jac, "RelTol", 1e-3);
%opts = odeset( 'RelTol', 1e-3);
rhs = @(t,y)neel_odesys(t,y,0, Bb, m_offset, m_b3, m_bp, m_bm,tau_N);
[t,y] = ode15s(rhs,t_vec , y0, opts);
end
