export rotz, simulationMNP

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
  B_ = p.B(t);
  B_1 = B_[1];
  B_2 = B_[2];
  B_3 = B_[3];

  neel_odesys_inner(y_out, y, p, B_1, B_2, B_3)

  return 
end

function neel_odesys_inner(y_out, y, p, B_1, B_2, B_3)
  y_out .= 0.0
  mul!(p.ytmp, p.m_bm, y)
  y_out .+= (B_1 - 1im*B_2) .* p.ytmp
  mul!(p.ytmp, p.m_bp, y) 
  y_out .+= (B_1 + 1im*B_2) .* p.ytmp
  mul!(p.ytmp, p.m_b3, y)
  y_out .+= B_3 .* p.ytmp
  mul!(p.ytmp, p.m_offset, y)
  y_out .+= p.ytmp

  # M = p.m_offset + B_3 .* p.m_b3 + (B_1 + 1im*B_2) .* p.m_bp + (B_1 - 1im*B_2) .* p.m_bm;
  # y_out .= M*y

  return 
end



function neel_odesys_jac(y_out, y, p, t)
  B_ = p.B(t);
  B_1 = B_[1];
  B_2 = B_[2];
  B_3 = B_[3];

  
  neel_odesys_jac_inner(y_out, y, p, B_1, B_2, B_3)
  
  return 
end

function neel_odesys_jac_inner(y_out, y, p, B_1, B_2, B_3)

  # M = p.m_offset + B_3 .* p.m_b3 + (B_1 + 1im*B_2) .* p.m_bp + (B_1 - 1im*B_2).*p.m_bm;
  # y_out .= M
  
  
  y_out.nzval .= 0
  y_out.nzval[p.idx_bm] .+= (B_1 - 1im*B_2) .* p.m_bm.nzval
  y_out.nzval[p.idx_bp] .+= (B_1 + 1im*B_2) .* p.m_bp.nzval
  y_out.nzval[p.idx_b3] .+= B_3 .* p.m_b3.nzval
  y_out.nzval[p.idx_offset] .+= p.m_offset.nzval
  
  return 
end

function generateSparseMatrices(N, p1, p2, p3, p4, tauN)

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
        V[ind] = -1/(2*tauN)*r*(r+1) + p4*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = -1im * p3 * q*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<(N-1)
            I[ind] = counter+2*(r+1)+2*(r+2);
            J[ind] = counter;
            V[ind] = -p4*r*(r+q+1)*(r+q+2)/((2*r+3)*(2*r+5));
            ind = ind+1;
        end
        if r>1 && q>(-r+1)
            I[ind] = counter-2*r-2*(r-1);
            J[ind] = counter;
            V[ind] = p4*(r+1)*(r-q)*(r-q-1)/((2*r-3)*(2*r-1));
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -1im * p3 * q*(r+q+1)/(2*r+3);
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
        V[ind] = -1im/2.0 * p1 *2*q;
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = p2*(r+1)*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -p2*r*(r+q+1)/(2*r+3);
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
            V[ind] = -1im/2.0 * p1 *(r-q)*(r+q+1);
            ind = ind+1;
        end
        if q<(r-1)
            I[ind] = counter-2*r+1;
            J[ind] = counter;
            V[ind] = p2 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1)+1;
            J[ind] = counter;
            V[ind] = p2 * r*(r+q+1)*(r+q+2)/(4*r+6);
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
            V[ind] = -1im/2.0*p1;
            ind = ind+1;
        end
        if q>(-r+1)
            I[ind] = counter-2*r-1;
            J[ind] = counter;
            V[ind] = -p2 * (r+1)/(4*r-2);
            ind = ind+1;

        end
        if r<N
            I[ind] = counter+2*(r+1)-1;
            J[ind] = counter;
            V[ind] = -p2 * r/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bm = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bm)
  
  return m_offset, m_b3, m_bp, m_bm
end

function simulationMNP(Bb, t_vec;
                       n = [0.0;0.0;1.0], 
                       MS = 474000.0, DCore = 20e-9, 
                       temp = 293.0, α = 0.1, kAnis = 625,
                       N = 20,
                       reltol = 1e-3, abstol=1e-6)

  kB = 1.38064852e-23
  gamGyro = 1.75*10^11
  VCore = pi/6 * DCore^3
  tauN = MS*VCore/(kB*temp*gamGyro)*(1+α^2)/(2*α)

  p1 = gamGyro/(1+α^2);
  p2 = α*gamGyro/(1+α^2);
  p3 = 2*gamGyro/(1+α^2)*kAnis/MS;
  p4 = 2*α*gamGyro/(1+α^2)*kAnis/MS;

  m_offset, m_b3, m_bp, m_bm = generateSparseMatrices(N, p1, p2, p3, p4, tauN)

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
  Mzero[1, 1] = 1 # Not clear why

  idx_offset = getIdxInM(Mzero, tmp_off)
  idx_b3 = getIdxInM(Mzero, tmp_b3)
  idx_bp = getIdxInM(Mzero, tmp_bp)
  idx_bm = getIdxInM(Mzero, tmp_bm)

  rot = rotz(n)   # Rotation matrix that rotates n to the z axis
  irot = inv(rot) # Rotation matrix that rotates the z axis to n
  
  BbRot(t) = rot*Bb(t)
  
  p = NeelParams(BbRot, m_offset, m_b3, m_bp, m_bm, tauN, ytmp, idx_offset, idx_b3, idx_bp, idx_bm)

  ff = ODEFunction(neel_odesys, jac = neel_odesys_jac, jac_prototype = Mzero)
  prob = ODEProblem(ff, y0, (t_vec[1],t_vec[end]), p)

  @time sol = solve(prob, QNDF(), reltol=reltol,abstol=abstol)
  #@time sol = solve(prob, CVODE_BDF(), reltol=reltol,abstol=abstol)
  #@time sol = solve(prob, Rosenbrock23(autodiff=false), reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), dt=1e-3, reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false), dt=1e-3, reltol=reltol)
  #@time sol = solve(prob, TRBDF2(autodiff=false), dt=1e-3, reltol=reltol)
  #@time sol = solve(prob, Rodas5(autodiff=false), reltol=reltol)

  #@time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), reltol=1e-3)

  
  @show sol.destats

  y = zeros(ComplexF64, length(t_vec), (N+1)^2)

  for ti=1:length(t_vec)
    y[ti, :] = sol(t_vec[ti])
  end

  # Calculate expectation from spherical harmonics
  xexptemp = real((4*pi/3)*(.5*y[:,2]-y[:,4]));
  yexptemp = real(-1im*(4*pi/3)*(y[:,4]+.5*y[:,2]));
  zexptemp = real((4*pi/3)*y[:,3]);

  # Rotate the coordinate system back (solver only works for the z-axis as
  # the easy axis)
  xexp = irot[1,1]*xexptemp + irot[1,2]*yexptemp + irot[1,3]*zexptemp;
  yexp = irot[2,1]*xexptemp + irot[2,2]*yexptemp + irot[2,3]*zexptemp;
  zexp = irot[3,1]*xexptemp + irot[3,2]*yexptemp + irot[3,3]*zexptemp;

  return t_vec, cat(xexp, yexp, zexp, dims=2)
end


function getIdxInM(M, V)
  M.nzval .= -1.0
  V.nzval .= 2.0
  M .+= V
  idx = findall(a-> real(a) > 0, M.nzval)
  return idx
end
