function out = isValidIndex(counter,dM, dL, M, L)
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

try
    Lnew = L(counter+dc);
    Mnew = M(counter+dc);
catch
    out = false;
    return
end

if Lnew == l+dL && Mnew == m+dM
    out = true;
    return
else
    out = false;
    return
end



end

