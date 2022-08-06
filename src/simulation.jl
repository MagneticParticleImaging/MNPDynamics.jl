export rotz, simulationMNP, simulationMNPMultiOffsets

struct MNPSimulationParams
  B::Function
  m_offset::SparseMatrixCSC{Complex{Float64},Int64}
  m_b3::SparseMatrixCSC{Complex{Float64},Int64}
  m_bp::SparseMatrixCSC{Complex{Float64},Int64}
  m_bm::SparseMatrixCSC{Complex{Float64},Int64}
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

function getIdxInM(M, V)
  M.nzval .= -1.0
  V.nzval .= 2.0
  M .+= V
  idx = findall(a-> real(a) > 0, M.nzval)
  return idx
end

###############


"""
  simulationMNP(B, t, offsets; kargs...) 
"""
function simulationMNP(Bb::g, tVec;
                       relaxation::RelaxationType = NEEL,
                       n = [0.0;0.0;1.0], 
                       MS = 474000.0, 
                       DCore = 20e-9, DHydro = DCore,
                       temp = 293.0, α = 0.1, kAnis = 625,
                       η = 1e-3,
                       N = 20,
                       tWarmup = 0.00005,
                       reltol = 1e-3, abstol=1e-6) where g

  kB = 1.38064852e-23
  gamGyro = 1.75*10^11
  VCore = pi/6 * DCore^3
  VHydro =  pi/6 * DHydro^3
  
  

  if relaxation == NEEL
    τNeel = MS*VCore/(kB*temp*gamGyro)*(1+α^2)/(2*α)
    p1 = gamGyro/(1+α^2);
    p2 = α*gamGyro/(1+α^2);
    p3 = 2*gamGyro/(1+α^2)*kAnis/MS;
    p4 = 2*α*gamGyro/(1+α^2)*kAnis/MS;

    rot = rotz(n)   # Rotation matrix that rotates n to the z axis
    irot = inv(rot) # Rotation matrix that rotates the z axis to n
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesNeel(N, p1, p2, p3, p4, τNeel)
  elseif relaxation == BROWN
    τBrown = 3*η*VHydro/(kB*temp)
    p2 = MS*VCore/(6*η*VHydro);

    rot = diagm([1,1,1])
    irot = diagm([1,1,1])
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesBrown(N, p2, τBrown)
  else
    error("Parameter relaxation needs to be either NEEL or BROWN!")
  end

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
  
  BbRot(t) = rot*Bb(t)
  
  p = MNPSimulationParams(BbRot, m_offset, m_b3, m_bp, m_bm, ytmp, idx_offset, idx_b3, idx_bp, idx_bm)

  ff = ODEFunction(neel_odesys, jac = neel_odesys_jac, jac_prototype = Mzero)
  prob = ODEProblem(ff, y0, (tVec[1]-tWarmup, tVec[end]), p)

  #@time 
  #sol = solve(prob, QNDF(), reltol=reltol, abstol=abstol)
  sol = solve(prob, FBDF(), reltol=reltol, abstol=abstol)
  #sol = solve(prob, Rodas5(autodiff=false), reltol=reltol)

  #@time sol = solve(prob, CVODE_BDF(), reltol=reltol,abstol=abstol)
  #@time sol = solve(prob, Rosenbrock23(autodiff=false), reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), dt=1e-3, reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false), dt=1e-3, reltol=reltol)
  #@time sol = solve(prob, TRBDF2(autodiff=false), dt=1e-3, reltol=reltol)
  #@time 


  #@time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), reltol=1e-3)

  
  #@show sol.destats

  y = zeros(ComplexF64, length(tVec), (N+1)^2)

  for ti=1:length(tVec)
    y[ti, :] = sol(tVec[ti])
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

  return tVec, cat(xexp, yexp, zexp, dims=2)
end


##################################

"""
  simulationMNPMultiOffsets(B, t, offsets; kargs...) 
"""
function simulationMNPMultiOffsets(B::g, t, offsets::Vector{NTuple{3,Float64}}; kargs...) where g 

  M = size(offsets,1)

  magnetizations = SharedArray{Float64}(length(t), 3, M)

  @sync @showprogress @distributed for m=1:M
      B_ = t -> ( B(t) .+ offsets[m] )
      t, y = simulationMNP(B_, t; kargs...)

      magnetizations[:,:,m] .= y
  end

  return magnetizations
end