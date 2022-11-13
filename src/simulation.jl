export rotz, simulationMNP, simulationMNPMultiParams

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

# forward declaration
function simulationMNPFokkerPlank end

"""
  simulationMNP(B, t, offsets; kargs...) 
"""
function simulationMNP(B::g, t; kargs...) where g
  if haskey(kargs, :neuralNetwork)
    return simulationMNPNeuralNetwork(B, t, kargs[:neuralNetwork]; kargs...)
  else
    return simulationMNPFokkerPlank(B, t; kargs...)
  end
end


function simulationMNPFokkerPlank(B::g, tVec;
                       relaxation::RelaxationType = NEEL, 
                       MS = 474000.0, 
                       DCore = 20e-9, DHydro = DCore,
                       temp = 293.0, α = 0.1, kAnis = 625,
                       η = 1e-3,
                       N = 20,
                       tWarmup = 0.00005,
                       solver = :FBDF,
                       derivative = false,
                       reltol = 1e-3, abstol=1e-6) where g

  if typeof(kAnis) <: AbstractVector
    kAnis_ = norm(kAnis)
    n = norm(kAnis) > 0 ? normalize(kAnis) : [1.0, 0.0, 0.0]
  else
    kAnis_ = kAnis
    n = [0.0;0.0;1.0]
  end

  kB = 1.38064852e-23
  gamGyro = 1.75*10^11
  VCore = pi/6 * DCore^3
  VHydro =  pi/6 * DHydro^3
  

  if relaxation == NEEL
    τNeel = MS*VCore/(kB*temp*gamGyro)*(1+α^2)/(2*α)
    p1 = gamGyro/(1+α^2);
    p2 = α*gamGyro/(1+α^2);
    p3 = 2*gamGyro/(1+α^2)*kAnis_/MS;
    p4 = 2*α*gamGyro/(1+α^2)*kAnis_/MS;

    if N == 0
      N = max(10, round(Int, 10+kAnis_/5000*30) )
    end

    rot = rotz(n)   # Rotation matrix that rotates n to the z axis
    irot = inv(rot) # Rotation matrix that rotates the z axis to n
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesNeel(N, p1, p2, p3, p4, τNeel)
  elseif relaxation == BROWN
    τBrown = 3*η*VHydro/(kB*temp)
    p2 = MS*VCore/(6*η*VHydro);

    rot = diagm([1,1,1])
    irot = diagm([1,1,1])
    m_offset, m_b3, m_bp, m_bm = generateSparseMatricesBrown(N, p2, τBrown)
  elseif relaxation == NO_RELAXATION
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
  
  BRot(t) = rot*B(t)
  
  p = MNPSimulationParams(BRot, m_offset, m_b3, m_bp, m_bm, ytmp, idx_offset, idx_b3, idx_bp, idx_bm)

  ff = ODEFunction(neel_odesys, jac = neel_odesys_jac, jac_prototype = Mzero)
  dt = tVec[2] - tVec[1]
  prob = ODEProblem(ff, y0, (tVec[1]-tWarmup, tVec[end] + dt), p)

  # The following tries to find out discontinuities which helps the solver
  B_ = [B(t)[d] for d=1:3, t in tVec]
  BD_ = vec(sum(abs.(diff(B_, dims=2)),dims=1) ./ maximum(abs.(B_)))
  tstops = ((tVec[1:end-1])[BD_ .> 0.2]) # 0.2 is a magic number

  #@time 
  #sol = solve(prob, QNDF(), reltol=reltol, abstol=abstol)
  if solver == :FBDF
    sol = solve(prob, FBDF(), reltol=reltol, abstol=abstol, tstops=tstops)#, tstops=tVec)

    #choice_function(integrator) = (Int(integrator.dt<dt/10) + 1)
    #alg_switch = CompositeAlgorithm((FBDF(), Rodas5(autodiff=false)), choice_function)
    ##alg_switch = AutoSwitch(FBDF(), Rodas5(autodiff=false))
    ##alg_switch = AutoTsit5(Rosenbrock23(autodiff=false))
    #sol = solve(prob, alg_switch, reltol=reltol, abstol=abstol)

  elseif solver == :Rodas5
    sol = solve(prob, Rodas5(autodiff=false), reltol=reltol, abstol=abstol, tstops=tstops)
  else
    error("Solver $(solver) not available")
  end

  #@time sol = solve(prob, CVODE_BDF(), reltol=reltol,abstol=abstol)
  #@time sol = solve(prob, Rosenbrock23(autodiff=false), reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), dt=1e-3, reltol=reltol)
 # @time sol = solve(prob, Rodas5(autodiff=false), dt=1e-3, reltol=reltol)
  #@time sol = solve(prob, TRBDF2(autodiff=false), dt=1e-3, reltol=reltol)
  #@time 


  #@time sol = solve(prob, Rodas5(autodiff=false, linsolve=KLUFactorization(reuse_symbolic=false)), reltol=1e-3)

  
  #@show sol.destats

  y = zeros(ComplexF64, length(tVec), (N+1)^2)

  if !derivative
    for ti=1:length(tVec)
      y[ti, :] = sol(tVec[ti])
    end
  else
    for ti=1:length(tVec)
      y[ti, :] = (sol(tVec[ti]+eps()) - sol(tVec[ti])) / eps()
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


"""
    simulationMNP(B::Matrix{T}, t; kargs...) 

This version takes the fields in discretized form
"""
function simulationMNP(B::Matrix{T}, t; kargs...) where {T}

  function fieldInterpolator(field, t)
    function help_(field, t, d)
      itp = interpolate(field[:,d], BSpline(Cubic(Flat(OnCell()))))
      etp = extrapolate(itp, Flat())
      sitp = scale(etp, t)
      return sitp
    end
    return (help_(field,t,1), help_(field,t,2), help_(field,t,3))
  end

  magneticMoments = zeros(Float64, length(t), 3)

  let param = fieldInterpolator(B, t)
    BFunc = (t) -> ( [param[1](t), param[2](t), param[3](t)] )
    magneticMoments[:,:] .= simulationMNP(BFunc, t; kargs...)
  end
  return magneticMoments
end




##################################

"""
  simulationMNPMultiParams(B, t, params; kargs...) 
"""
function simulationMNPMultiParams(B::G, t, params::Vector{P}; kargs...) where {G,P}
  numThreadsBLAS = BLAS.get_num_threads()
  M = length(params)

  magneticMoments = SharedArray{Float64}(length(t), 3, M)
  kargsInner = copy(kargs)

  #try
    BLAS.set_num_threads(1)
    @sync @showprogress @distributed for m=1:M
      let p=params[m]
        B_ = t -> ( B(t, p) )

        # this can be extended to more parameters
        if haskey(kargs, :kAnis) && typeof(kargs[:kAnis]) <: AbstractVector  && eltype(kargs[:kAnis]) <: AbstractVector
          kargsInner[:kAnis] = kargs[:kAnis][m]
        end

        if haskey(kargs, :DCore) && typeof(kargs[:DCore]) <: AbstractArray
          kargsInner[:DCore] = kargs[:DCore][m]
        end

        y = simulationMNP(B_, t; kargsInner...)
        magneticMoments[:,:,m] .= y
        GC.gc()
      end
    end
  #finally
    BLAS.set_num_threads(numThreadsBLAS)
  #end

  return magneticMoments
end


"""
  simulationMNPMultiParams(B::Vector{Matrix{T}}, t; kargs...) 

This version takes the fields in discretized form
"""
function simulationMNPMultiParams(B::Array{T,3}, t; kargs...) where {T}

  M = size(B,3)

  BFunc = (t, param) -> ( [param[1](t), param[2](t), param[3](t)] )

  function fieldInterpolator(field, t)
    function help_(field, t, d)
      itp = interpolate(field[:,d], BSpline(Cubic(Flat(OnCell()))))
      etp = extrapolate(itp, Flat())
      sitp = scale(etp, t)
      return sitp
    end
    return (help_(field,t,1), help_(field,t,2), help_(field,t,3))
  end

  params = vec([ fieldInterpolator(B[:,:,z], t)  for z=1:M ])

  return simulationMNPMultiParams(BFunc, t, params; kargs...)
end

function simulationMNPMultiParams(filename::AbstractString, B::Array{T,3}, t, params) where {T}
  if !isfile(filename)
    m = simulationMNPMultiParams(B, t; params...)
    h5open(filename,"w") do h5
      h5["magneticMoments"] = m
      h5["time"] = collect(t)
      h5["B"] = B
      h5["DCore"] = params[:DCore]
      h5["kAnis"] = eltype(params[:kAnis]) <: AbstractVector ? 
              [params[:kAnis][z][d] for d=1:3, z=1:length(params[:kAnis])]  : params[:kAnis]
    end
  else
    m, B = h5open(filename,"r") do h5
      m = read(h5["magneticMoments"])
      B = read(h5["B"])
      params[:DCore] = read(h5["DCore"])
      kAnis_ = read(h5["kAnis"])
      params[:kAnis] = [ collect(kAnis_[:,z]) for z=1:size(kAnis_, 2) ]
      (m, B)
    end
  end
  return m, B
end





export generateStructuredFields
function generateStructuredFields(params, t, Z; fieldType::FieldType, maxField, filterFactor=7)

  if fieldType == RANDOM_FIELD
    B = rand_interval(-1, 1, length(t), 3, Z)
    for z=1:Z
      for d=1:3
        B[:,d,z] = imfilter(B[:,d,z], Kernel.gaussian((filterFactor,))) 
        B[:,d,z] ./= maximum(abs.(B[:,d,z]))
        B[:,d,z] .= maxField*(rand()*B[:,d,z] ) #.+ rand_interval(-1,1)*ones(Float32,length(t))*0.5)
      end
    end
  elseif fieldType == HARMONIC_RANDOM_FIELD
    B = zeros(Float32, length(t), 3, Z)
    for z=1:Z
      for d=1:3
        γ = rand()
        f = rand_interval(20e3, 50e3)
        offset = rand_interval(-1,1)
        phase = rand_interval(-π,π)
        B[:,d,z] = maxField*(γ*sin.(2*π*f*t.+phase) .+ (1-γ)*offset)
      end
    end
  elseif fieldType == HARMONIC_MPI_FIELD
    B = zeros(Float32, length(t), 3, Z)
    freq = params[:frequencies]
    for z=1:Z
      for d=1:3
        B[:,d,z] = maxField*(sin.(2*pi*freq[d]*t) .+ (1-γ)*offset)
      end
    end    
  else
    error("field type $fieldType not supported!")
  end

  paramsInner = copy(params)

  if haskey(params, :DCore) && typeof(params[:DCore]) <: Tuple
    paramsInner[:DCore] = rand_interval(params[:DCore][1], params[:DCore][2], Z) 
    useDCore = true
  end

  if haskey(params, :kAnis) && typeof(params[:kAnis]) <: Tuple
    paramsInner[:kAnis] = [ rand_interval(params[:kAnis][1], params[:kAnis][2])*randAxis() for z=1:Z ]
    #paramsInner[:kAnis] = [ rand_interval(params[:kAnis][1], params[:kAnis][2])*[1,0,0] for z=1:Z ]

    useKAnis = true
  end

  return B, paramsInner
end