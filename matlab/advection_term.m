function [a] = advection_term(x,B,n, p1, p2, p3, p4)
% advection term for a fixed time t, location x and magnetic field B = B(t,x) 
B2 = (x'*n)'.*n;
Beff1 = p1*B + p3*B2;
Beff2 = p2*B + p4*B2;
%a = p1*cross(B.*ones(size(x)), x) + p2*(B - sum((B.*x),1).*x) + p3*cross(B2,x)...
    %+p4*(B2-sum((B2.*x),1).*x);
a = cross(Beff1, x) + (Beff2 - sum(Beff2.*x,1).*x);
end

