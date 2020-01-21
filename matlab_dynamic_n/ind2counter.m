function dc = ind2counter(counter, dM, dL, M,L)


m = M(counter);
l = L(counter);

dc = dM;
if dL == 1
    dc = dc + 2*(l+1);
elseif dL == 2
    dc = dc + 2*(l+1) + 2*(l+2);
elseif dL == -1
    dc = dc - 2*l;
elseif dL == -2
    dc = dc - 2*l - 2*(l-1);
end

end

