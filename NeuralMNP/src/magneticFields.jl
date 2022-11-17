
export generateStructuredFields
function generateStructuredFields(params, t, Z; fieldType::FieldType, maxField, filterFactor=7)

  if fieldType == RANDOM_FIELD
    B = rand_interval(-1, 1, length(t), 3, Z)
    for z=1:Z
      for d=1:3
        B[:,d,z] = imfilter(B[:,d,z], Kernel.gaussian((filterFactor,))) 
        B[:,d,z] ./= maximum(abs.(B[:,d,z]))
        B[:,d,z] .= maxField*(rand()*B[:,d,z] ) .+ 0.5*maxField*rand_interval(-1,1)*ones(Float32,length(t))
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