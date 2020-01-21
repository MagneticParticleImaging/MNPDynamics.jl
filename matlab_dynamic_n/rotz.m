function [R] = rotz(x)
%returns the rotation matrix that rotates the vector x onto the z axis.

x = x/norm(x);
n = cross(x,[0;0;1]);
if norm(n)>0
    n = n/norm(n);
else
    R = eye(3);
    return
end
a = acos(x(3));
n1 = n(1);
n2 = n(2);
n3 = n(3);
R = [n1^2*(1-cos(a)) + cos(a), n1*n2*(1-cos(a))-n3*sin(a),n1*n3*(1-cos(a))+n2*sin(a);...
    n2*n1*(1-cos(a)) + n3*sin(a), n2^2*(1-cos(a))+cos(a), n2*n3*(1-cos(a))-n1*sin(a);...
    n3*n1*(1-cos(a))-n2*sin(a), n3*n2*(1-cos(a))+n1*sin(a), n3^2*(1-cos(a))+cos(a)];


end

