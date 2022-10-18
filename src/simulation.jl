export rotz, simulationMNP, simulationMNPMultiParams

struct MNPSimulationParams
  B::Function
  m_offset::SparseMatrixCSC{ComplexF64,Int64}
  m_b3::SparseMatrixCSC{ComplexF64,Int64}
  m_bp::SparseMatrixCSC{ComplexF64,Int64}
  m_bm::SparseMatrixCSC{ComplexF64,Int64}
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
  # M = p.m_offset + B_3 .* p.m_b3 + (B_1 + 1im*B_2) .* p.m_bp + (B_1 - 1im*B_2) .* p.m_bm;
  # y_out .= M*y
  mul!(y_out, p.m_bm, y, B_1 - im*B_2, 0)
  mul!(y_out, p.m_bp, y, (B_1 + im*B_2), 1) 
  mul!(y_out, p.m_b3, y, B_3, 1)
  mul!(y_out, p.m_offset, y, 1.0, 1)
  
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
  # y_out.nzval[p.idx_bm] .+= (B_1 - im*B_2) .* p.m_bm.nzval
  @inbounds for (i,j) in enumerate(p.idx_bm)
    y_out.nzval[j] += (B_1 - im*B_2) * p.m_bm.nzval[i]
  end
  # y_out.nzval[p.idx_bp] .+= (B_1 + im*B_2) .* p.m_bp.nzval
  @inbounds for (i,j) in enumerate(p.idx_bp)
    y_out.nzval[j] += (B_1 + im*B_2) * p.m_bp.nzval[i]
  end
  # y_out.nzval[p.idx_b3] .+= B_3 .* p.m_b3.nzval
  @inbounds for (i,j) in enumerate(p.idx_b3)
    y_out.nzval[j] += B_3 * p.m_b3.nzval[i]
  end
  # y_out.nzval[p.idx_offset] .+= p.m_offset.nzval
  @inbounds for (i,j) in enumerate(p.idx_offset)
    y_out.nzval[j] += p.m_offset.nzval[i]
  end

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
function simulationMNP(B::g, tVec;
                       relaxation::RelaxationType = NEEL,
                       n = [0.0;0.0;1.0], 
                       MS = 474000.0, 
                       DCore = 20e-9, 
                       DHydro = DCore,
                       temp = 293.0,
                       α = 0.1, 
                       kAnis = 625,
                       η = 1e-3,
                       N = 20,
                       tWarmup = 0.00005,
                       solver = :FBDF,
                       reltol = 1e-3, 
                       abstol=1e-6,
                       derivative = false,
                       derivative_order=50 # order of finite differentiation method
                       ) where g
    
  if relaxation == NO_RELAXATION
    y = zeros(Float64, length(tVec), 3)
                    
    if !derivative
      for ti=1:length(tVec)
        y[ti, :] = langevin(B(tVec[ti]); DCore, temp, MS)
      end
    else
      for ti=1:length(tVec)
        y[ti, :] = (langevin(B(tVec[ti]+eps()); DCore, temp, MS)-langevin(B(tVec[ti]); DCore, temp, MS)) / eps()
      end
    end
                    
    return y
  elseif relaxation == NEEL || relaxation == BROWN
    prob, rot = set_up_simulation(B, tVec; 
                            relaxation=relaxation, n=n, MS=MS, DCore=DCore, 
                            DHydro=DHydro, temp=temp, α=α, kAnis=kAnis, η=η, 
                            N=N, tWarmup=tWarmup, derivative_order=derivative_order)
    sol = simulate(prob, solver, reltol, abstol)
    sol_sampled = sample_solution(sol, tVec, rot, N, derivative, derivative_order)
    return sol_sampled
  else
    error("Relaxation type unknown!")
  end
end

function set_up_simulation(B::g, tVec;
                    relaxation::RelaxationType = NEEL,
                    n = [0.0;0.0;1.0], 
                    MS = 474000.0, 
                    DCore = 20e-9, 
                    DHydro = DCore,
                    temp = 293.0, 
                    α = 0.1, 
                    kAnis = 625,
                    η = 1e-3,
                    N = 20,
                    tWarmup = 0.00005,
                    derivative_order = 50 # order of finite differentiation method
                    ) where g                  
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
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesNeel(N, p1, p2, p3, p4, τNeel)
  else relaxation == BROWN
    τBrown = 3*η*VHydro/(kB*temp)
    p2 = MS*VCore/(6*η*VHydro);

    rot = diagm([1,1,1])
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesBrown(N, p2, τBrown)
  end

  # initial value
  y0 = zeros(ComplexF64, (N+1)^2);
  y0[1] = 1/(4*pi);

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
  
  BRot = (t) -> rot*B(t)
  
  p = MNPSimulationParams(BRot, m_offset, m_b3, m_bp, m_bm, idx_offset, idx_b3, idx_bp, idx_bm)

  ff = ODEFunction(neel_odesys, jac = neel_odesys_jac, jac_prototype = Mzero)
  dt = tVec[2] - tVec[1]
  prob = ODEProblem(ff, y0, (tVec[1] - derivative_order*dt - tWarmup, tVec[end] + derivative_order*dt), p)
  return prob, rot
end


function simulate(prob, solver, reltol, abstol)
  if solver == :FBDF
    sol = solve(prob, FBDF(), reltol=reltol, abstol=abstol)
  elseif solver == :Rodas5
    sol = solve(prob, Rodas5(autodiff=false), reltol=reltol, abstol=abstol)
  else
    error("Solver $(solver) not available")
  end
  return sol
end


function sample_solution(sol, tVec, rot, N, derivative, derivative_order)
  y = zeros(ComplexF64, length(tVec), (N+1)^2)
  irot = inv(rot)

  if !derivative
    for (i,t) in enumerate(tVec)
      y[i, :] = sol(t)
    end
  else
    if typeof(tVec)<:StepRangeLen
      # New finite difference formulas for numerical differentiation
      # https://doi.org/10.1016/S0377-0427(99)00358-1
      # more accurate for highly oscilating functions than standard finite differences
      L = length(tVec)
      t_0 = first(tVec)
      h = step(tVec)
      n = derivative_order

      # sample solution at points in between the ones we aim to have
      timepointsn = range(start=t_0-(2*n-1)/2*h,step=h, length=L+2*n-1)
      yn = zeros(ComplexF64, length(timepointsn), (N+1)^2)
      for (i,t) in enumerate(timepointsn)
        yn[i,:] .= sol(t)
      end
      # finite difference formula is convolution like
      # define corresponding kernel to act on the time dimension only
      imkerneln = OffsetArray(reshape(vcat([-e(n,i) for i=n:-1:1],[e(n,i) for i=1:n])/h,2*n,1),-n:n-1,0:0)
      # apply finite difference formula and retrieve points corresponding to time points in tVec
      y .= imfilter(yn, imkerneln, ImageFiltering.Algorithm.FIR())[n+1:end-n+1,:]
    else
      for (i,t) in enumerate(tVec)
        # accuracy limited to 1e-4 to 1e-5 for oscilating functions up to 1.25 MHz
        y[i, :] = (sol(t+eps()) - sol(t)) / eps()
      end 
    end
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

  return cat(xexp, yexp, zexp, dims=2)
end






##################################

"""
  simulationMNPMultiParams(B, t, params; kargs...) 
"""
function simulationMNPMultiParams(B::G, t, params::Vector{P}; kargs...) where {G,P}

  M = size(params,1)

  magnetizations = SharedArray{Float64}(length(t), 3, M)
  kargsInner = copy(kargs)

  @sync @showprogress @distributed for m=1:M
    let p=params[m]
      B_ = t -> ( B(t, p) )

      # this can be extended to more parameters
      if haskey(kargs, :n) && eltype(kargs[:n]) <: Tuple
        n = [kargs[:n][m]...]
        kargsInner[:n] = norm(n) > 0 ? n ./ norm(n) : [1,0,0]
        kargsInner[:kAnis] = norm(n)*kargs[:kAnis]
      end

      y = simulationMNP(B_, t; kargsInner...)
      magnetizations[:,:,m] .= y
    end
  end

  return magnetizations
end