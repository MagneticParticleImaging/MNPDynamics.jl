function dydt = odesys(t,y, jacflag, B, n, m_offset, m_b3, m_bp, m_bm,m_squ,m_nmin3,m_npin3,m_nmi,m_npi,tau_N)
% Returns the right-hand side of the ODE system for given time t
% and given state y.
%INPUTS:
%   t: a single scalar time value
%   y: state y for which the right-hand side of the ODE is to be evaluated
%   jacflag: if 1, return the matrix (the jacobian of the rhs), if 0 return
%   the matrix applied to y
%   B: function that represents the magnetic field
%   m_offset, m_b3, m_bp, m_bm: Matrices that are independent of time and
%   are calculated beforehand.
%   tau_N: Neel relaxation time constant.
B = B(t);
B_1 = B(1,:);
B_2 = B(2,:);
B_3 = B(3,:);
if isa(n, 'function_handle')
    n = n(t);
end
if length(n)~=1
    n1 = n(1);
    n2 = n(2);
    n3 = n(3);
end

if n==0
    M = m_offset + B_3.*m_b3 + (B_1+1i*B_2).*m_bp + (B_1-1i*B_2).*m_bm;
else
    M =  m_offset + B_3 .* m_b3 + (B_1 + 1i*B_2) .* m_bp + (B_1 - 1i*B_2).*m_bm...
        +(n1^2+n2^2-2*n3^2).*m_squ + (n1-1i*n2)*n3.*m_nmin3 + (n1+1i*n2)*n3.*m_npin3...
        +(n1-1i*n2)^2.*m_nmi + (n1+1i*n2)^2.*m_npi;
end



if jacflag
    dydt = M;
else
    dydt = M*y;
end

end

