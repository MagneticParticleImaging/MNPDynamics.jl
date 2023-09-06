
"""
  simulationMNPMultiParams(B, t, params; kargs...) 
"""
function simulationMNPMultiParams(B::G, t, params::Vector{P}; kargs...) where {G,P}
  M = length(params)
  if haskey(kargs, :neuralNetwork)

    magneticMoments = zeros(Float32, length(t), 3, M)

    kargsInner = copy(kargs)

    @showprogress for m=1:M
      B_ = t -> ( B(t, params[m]) )

      # this can be extended to more parameters
      if haskey(kargs, :kAnis) && typeof(kargs[:kAnis]) <: AbstractVector &&
              ( eltype(kargs[:kAnis]) <: AbstractVector || eltype(kargs[:kAnis]) <: Tuple )
        kargsInner[:kAnis] = kargs[:kAnis][m]
      end

      if haskey(kargs, :DCore) && typeof(kargs[:DCore]) <: AbstractArray
        kargsInner[:DCore] = kargs[:DCore][m]
      end

      y = simulationMNP(B_, t, kargs[:alg]; kargsInner...)
      magneticMoments[:,:,m] .= y

      #params[:kAnis] = anisotropyAxis[z]
      #sm[:,:,z] .= simulationMNP(t -> BSM(t, offsets[z]), tSM, NeuralNetworkMNP; device, params...)
    end
  
    return magneticMoments
  else
    numThreadsBLAS = BLAS.get_num_threads()

    magneticMoments = SharedArray{Float64}(length(t), 3, M)
    kargsInner = copy(kargs)
    if haskey(kargs, :DCore) && typeof(kargs[:DCore]) <: AbstractArray
      delete!(kargsInner, :DCore)
    end
    if haskey(kargs, :kAnis) && typeof(kargs[:kAnis]) <: AbstractVector &&
                      ( eltype(kargs[:kAnis]) <: AbstractVector ||     eltype(kargs[:kAnis]) <: Tuple )
      delete!(kargsInner, :kAnis)
    end

    #prog = Progress(M, 1, "Simulation")
    #try
      BLAS.set_num_threads(1)

      #for m=1:M
      #Threads.@threads for m=1:M
      @sync @showprogress @distributed for m=1:M
        let p=params[m], kargsInner=copy(kargsInner)
          B_ = t -> ( B(t, p) )

          # this can be extended to more parameters
          if haskey(kargs, :kAnis) && typeof(kargs[:kAnis]) <: AbstractVector &&
                  ( eltype(kargs[:kAnis]) <: AbstractVector || eltype(kargs[:kAnis]) <: Tuple )
            kargsInner[:kAnis] = kargs[:kAnis][m]
          end

          if haskey(kargs, :DCore) && typeof(kargs[:DCore]) <: AbstractArray
            kargsInner[:DCore] = kargs[:DCore][m]
          end

          y = simulationMNP(B_, t; kargsInner...)
          magneticMoments[:,:,m] .= y
          GC.gc()
        end
        #next!(prog)
      end
    #finally
      BLAS.set_num_threads(numThreadsBLAS)
    #end

    return Array(magneticMoments)
  end
end


"""
  simulationMNPMultiParams(B::Vector{Matrix{T}}, t; kargs...) 

This version takes the fields in discretized form
"""
function simulationMNPMultiParams(B::Array{T,3}, t; kargs...) where {T}

  M = size(B,3)

  BFunc = (t, param) -> ( SVector{3,T}(param[1](t), param[2](t), param[3](t)) )

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

function simulationMNPMultiParams(filename::AbstractString, B::Array{T,3}, t, params; force=false) where {T}
  if !isfile(filename) || force
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