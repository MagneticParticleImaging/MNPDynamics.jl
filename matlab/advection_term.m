function [a] = advection_term(x,B,n, p1, p2, p3, p4)
% advection term for a fixed time t, location x and magnetic field B = B(t,x)
B2 = (x'*n)'.*n;
if (p1~=0 || p3 ~= 0)
    Beff1 = p1*B + p3*B2;
end
Beff2 = p2*B + p4*B2;

if (p1 ~=0 || p3 ~= 0)
    a = cross(Beff1, x) + (Beff2 - sum(Beff2.*x,1).*x);
else
    a = Beff2 - sum(Beff2.*x,1).*x;
end
end

