function rotz(x)
  #returns the rotation matrix that rotates the vector x onto the z axis.

  x = x/norm(x);
  n = cross(x,[0;0;1]);
  if norm(n)>0
    n = n/norm(n);
  else
    R = Matrix{Float64}(I, 3, 3)
    return R
  end
  a = acos(x[3]);
  n1 = n[1];
  n2 = n[2];
  n3 = n[3];
  R = [n1^2*(1-cos(a))+cos(a)  n1*n2*(1-cos(a))-n3*sin(a)  n1*n3*(1-cos(a))+n2*sin(a);
    n2*n1*(1-cos(a))+n3*sin(a)  n2^2*(1-cos(a))+cos(a)  n2*n3*(1-cos(a))-n1*sin(a);
    n3*n1*(1-cos(a))-n2*sin(a)  n3*n2*(1-cos(a))+n1*sin(a)  n3^2*(1-cos(a))+cos(a)];

  return R
end

# New finite difference formulas for numerical differentiation
# https://doi.org/10.1016/S0377-0427(99)00358-1
function e(n,k)
  k = BigInt(k)
  n = BigInt(n)
  return Float64(((-1)^Int(k+1)*prod(2*n-1:-2:1)^2)//(2^(2*n-2)*factorial(n+k-1)*factorial(n-k)*(2*k-1)^2))
end


function langevin(H::V; DCore=25e-9, temp=294, MS=0.6/(4*π*1e-7)) where {T,V<:AbstractVector{T}}
  kB = 1.380650424e-23 #Boltzman constant
  μ₀ = 4*π*1e-7 #vacuum permeability
  msat = MS * π/6*DCore^3 #saturation magnetic moment of a single nanoparticle
  beta::T = msat /(kB*temp) #H measured in T/μ₀

  if norm(H)!=0
    x = beta*norm(H)
    return  (coth(x) - 1/x)*normalize(H) # msat*
  else
    return zeros(T,3)
  end  
end
