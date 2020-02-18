function [M] = FV_matrix(t, B, N, p1, p2, p3, p4, beta, mids, ds, n, iis, vals, A, C, a_ijs, e_is, areasidil, tr2edge, flow_signs)
% outputs the right-hand matrix for a given x and B(t,x).

B = B(t);
if isa(n, 'function_handle')
    n = n(t);
end

d_ijt = sum(advection_term(mids, B, n, p1, p2, p3, p4).*e_is,1)'.*ds;
d_ij = flow_signs.*d_ijt(tr2edge);

b_ij = (d_ij > 0);

a_ij = beta*b_ij + (1-beta)*a_ijs(:);

d_ij = areasidil.*d_ij;
d_ij1 = d_ij(1:N);
d_ij2 = d_ij(N+1:2*N);
d_ij3 = d_ij(2*N+1:end);

a_ij1 = a_ij(1:N);
a_ij2 = a_ij(N+1:2*N);
a_ij3 = a_ij(2*N+1:end);

vals(1:4:end-3) = (d_ij1.*a_ij1 + d_ij2.*a_ij2 + d_ij3.*a_ij3);
vals(2:4:end-2) = (1-a_ij1).*d_ij1;
vals(3:4:end-1) = (1-a_ij2).*d_ij2;
vals(4:4:end) = (1-a_ij3).*d_ij3;

A(iis) = vals;

M = C-A;

end

