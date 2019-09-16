export rotz, simulation_neel


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


struct NeelParams
  B::Function
  m_offset::SparseMatrixCSC{Complex{Float64},Int64}
  m_b3::SparseMatrixCSC{Complex{Float64},Int64}
  m_bp::SparseMatrixCSC{Complex{Float64},Int64}
  m_bm::SparseMatrixCSC{Complex{Float64},Int64}
  tau_N::Float64
  ytmp::Vector{ComplexF64}
  idx_offset::Vector{Int64}
  idx_b3::Vector{Int64}
  idx_bp::Vector{Int64}
  idx_bm::Vector{Int64}
end

function neel_odesys(y_out, y, p, t)
#@time begin
  B_ = p.B(t);
  B_1 = B_[1];
  B_2 = B_[2];
  B_3 = B_[3];

  #M = p.m_offset + B_3 .* p.m_b3 + (B_1 + 1im*B_2) .* p.m_bp + (B_1 - 1im*B_2) .* p.m_bm;
  #dydt = (1/(2*p.tau_N)) * M*y;
  
  y_out .= 0.0
  mul!(p.ytmp, p.m_bm, y)
  factor1 = (1.0/(2.0*p.tau_N)) *(B_1 - 1im*B_2) 
  y_out .+= factor1 .* p.ytmp
  mul!(p.ytmp, p.m_bp, y)
  factor2 = (1.0/(2.0*p.tau_N)) *(B_1 + 1im*B_2) 
  y_out .+= factor2 .* p.ytmp
  mul!(p.ytmp, p.m_b3, y)
  factor3 = (1.0/(2.0*p.tau_N)) * B_3
  y_out .+= factor3 .* p.ytmp
  mul!(p.ytmp, p.m_offset, y)
  factor4 = (1.0/(2.0*p.tau_N))
  y_out .+= factor4 .* p.ytmp
  
  #y_out .= dydt

#end
  return 
end


function neel_odesys_jac(y_out, y, p, t)
#@time begin
  B_ = p.B(t);
  B_1 = B_[1];
  B_2 = B_[2];
  B_3 = B_[3];

  #=
  M = p.m_offset + B_3 .* p.m_b3 + (B_1 + 1im*B_2) .* p.m_bp + (B_1 - 1im*B_2).*p.m_bm;
  dydt = (1/(2*p.tau_N))*M;
  y_out .= dydt
  =#
  
  y_out.nzval .= 0
  factor1 = (1.0/(2.0*p.tau_N)) *(B_1 - 1im*B_2)
  y_out.nzval[p.idx_bm] .+= factor1 .* p.m_bm.nzval
  factor2 = (1.0/(2.0*p.tau_N)) *(B_1 + 1im*B_2)
  y_out.nzval[p.idx_bp] .+= factor2 .* p.m_bp.nzval
  factor3 = (1.0/(2.0*p.tau_N)) * B_3
  y_out.nzval[p.idx_b3] .+= factor3 .* p.m_b3.nzval
  factor4 = (1.0/(2.0*p.tau_N))
  y_out.nzval[p.idx_offset] .+= factor4 .* p.m_offset.nzval
  
  #end
  return 
end


function simulation_neel(Bb, pr1, pr2, tau_N, alpha, t_vec, N)
  counter = 0;
  nz = 0;
  for r=0:N
    for q=-r:r
        counter = counter+1;

        nz = nz+1;
        if q!=-r
            nz = nz+1;
            if r!=0 && q!=r && q!=-r
                nz = nz+1;
            end
        end
        if q!=r
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

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        I[ind] = counter;
        J[ind] = counter;
        V[ind] = -r*(r+1) + pr2* 2*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = 2*1im/alpha * pr2 * q*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<(N-1)
            I[ind] = counter+2*(r+1)+2*(r+2);
            J[ind] = counter;
            V[ind] = -pr2 * 2*r*(r+q+1)*(r+q+2)/((2*r+3)*(2*r+5));
            ind = ind+1;
        end
        if r>1 && q>(-r+1)
            I[ind] = counter-2*r-2*(r-1);
            J[ind] = counter;
            V[ind] = pr2 * 2*(r+1)*(r-q)*(r-q-1)/((2*r-3)*(2*r-1));
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = 2*1im/alpha * pr2 * q*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1

  m_offset = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_offset)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        I[ind] = counter;
        J[ind] = counter;
        V[ind] = 1im/(2*alpha) * pr1 *2*q;
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = pr1*(r+1)*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -pr1*r*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_b3 = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_b3)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q!=r
            I[ind] = counter+1;
            J[ind] = counter;
            V[ind] = 1im/(2*alpha) * pr1 *(r-q)*(r+q+1);
            ind = ind+1;
        end
        if q<(r-1)
            I[ind] = counter-2*r+1;
            J[ind] = counter;
            V[ind] = pr1 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1)+1;
            J[ind] = counter;
            V[ind] = pr1 * r*(r+q+1)*(r+q+2)/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bp = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bp)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q!=-r
            I[ind] = counter-1;
            J[ind] = counter;
            V[ind] = 1im/(2*alpha)*pr1;
            ind = ind+1;
        end
        if q>(-r+1)
            I[ind] = counter-2*r-1;
            J[ind] = counter;
            V[ind] = -pr1 * (r+1)/(4*r-2);
            ind = ind+1;

        end
        if r<N
            I[ind] = counter+2*(r+1)-1;
            J[ind] = counter;
            V[ind] = -pr1 * r/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bm = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bm)

  # initial value
  y0 = zeros(ComplexF64, (N+1)^2);
  y0[1] = 1/(4*pi);
  ytmp = zeros(ComplexF64, (N+1)^2);

  # calculate the indices occuring in the jacobian  
  tmp_off = deepcopy(m_offset)
  tmp_off.nzval .= 1
  tmp_b3 = deepcopy(m_b3)
  tmp_b3.nzval .= 1
  tmp_bp = deepcopy(m_bp)
  tmp_bp.nzval .= 1
  tmp_bm = deepcopy(m_bm)
  tmp_bm.nzval .= 1
  
  Mzero = tmp_off .+ tmp_b3 .+ tmp_bp .+ tmp_bm
  
  idx_offset = getIdxInM(Mzero, tmp_off)
  idx_b3 = getIdxInM(Mzero, tmp_b3)
  idx_bp = getIdxInM(Mzero, tmp_bp)
  idx_bm = getIdxInM(Mzero, tmp_bm)
  
  p = NeelParams(Bb, m_offset, m_b3, m_bp, m_bm, tau_N, ytmp, idx_offset, idx_b3, idx_bp, idx_bm)
  
  ff = ODEFunction(neel_odesys, jac = neel_odesys_jac, jac_prototype = Mzero)
  prob = ODEProblem(ff, y0, (t_vec[1],t_vec[end]), p)

  #@time sol = solve(prob, QNDF(), reltol=1e-3,abstol=1e-3)
  #@time sol = solve(prob, CVODE_BDF(), reltol=1e-3,abstol=1e-3)
  #@time sol = solve(prob, Rosenbrock23(autodiff=false), reltol=1e-3)
  #@time sol = solve(prob, Rodas5(autodiff=false), dt=1e-3, reltol=1e-3)
  @time sol = solve(prob, TRBDF2(autodiff=false), dt=1e-3, reltol=1e-3)
  #@time sol = solve(prob, Rodas5(autodiff=false), reltol=1e-3)
  
  @show sol.destats

  y = zeros(ComplexF64,length(t_vec), (N+1)^2)

  for ti=1:length(t_vec)
    y[ti, :] = sol(t_vec[ti])
  end
  return t_vec, y
end


function getIdxInM(M, V)
  M.nzval .= -1.0
  V.nzval .= 2.0
  M .+= V
  return findall(a-> real(a) > 0, M.nzval)
end
