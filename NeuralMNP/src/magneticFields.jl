
export generateStructuredFields
function generateStructuredFields(params, t, Z; fieldType::FieldType, 
                                             anisotropyAxis=nothing,
                                             dims = 1:3,
                                             freqInterval = (20e3, 50e3))

  filterFactor = get(params, :filterFactor, (4,20))
  maxField = params[:maxField]

  if fieldType == RANDOM_FIELD
    B = rand_interval(-1, 1, length(t), 3, Z)
    for z=1:Z
      for d in dims
        filtFactor = rand_interval(filterFactor[1], filterFactor[2])
        B[:,d,z] = imfilter(B[:,d,z], Kernel.gaussian((filtFactor,))) 
        B[:,d,z] ./= maximum(abs.(B[:,d,z]))
        B[:,d,z] .= maxField*(rand()*B[:,d,z])
      end
    end
  elseif fieldType == HARMONIC_RANDOM_FIELD
    B = zeros(Float32, length(t), 3, Z)
    for z=1:Z
      for d in dims
        γ = rand()
        f = rand_interval(freqInterval[1], freqInterval[2])
        offset = rand_interval(-1,1)
        phase = rand_interval(-π,π)
        B[:,d,z] = maxField*(γ*sin.(2*π*f*t.+phase) .+ (1-γ)*offset)
      end
    end
  elseif fieldType == HARMONIC_MPI_FIELD
    B = zeros(Float32, length(t), 3, Z)
    freq = params[:frequencies]
    for z=1:Z
      for d in dims
        B[:,d,z] = maxField*(sin.(2*pi*freq[d]*t) .+ (1-γ)*offset)
      end
    end    
  else
    error("field type $fieldType not supported!")
  end

  paramsInner = copy(params)

  if haskey(params, :DCore) && typeof(params[:DCore]) <: Tuple
    paramsInner[:DCore] = rand_interval(params[:DCore][1], params[:DCore][2], Z) 
  end

  if haskey(params, :kAnis) && typeof(params[:kAnis]) <: Tuple
    if anisotropyAxis == nothing
      paramsInner[:kAnis] = [ rand_interval(params[:kAnis][1], params[:kAnis][2])*randAxis() for z=1:Z ]
    else
      paramsInner[:kAnis] = [ rand_interval(params[:kAnis][1], params[:kAnis][2])*anisotropyAxis for z=1:Z ]
    end
  end

  return B, paramsInner
end

export combineFields
function combineFields(fields, params; shuffle=true)
  D = length(fields)
  Z = sum(ntuple(d->size(fields[d],3), D))
  B = zeros(Float32, size(fields[1],1), 3, Z)
  param = copy(params[1])
  z = 1
  for d=1:D
    B[:,:,z:z+size(fields[d],3)-1] = fields[d]
    z += size(fields[d],3)
  end
  for d=2:D
    append!(param[:DCore], params[d][:DCore])
    append!(param[:kAnis], params[d][:kAnis])
  end

  if shuffle
   idx = Random.shuffle(1:Z)
   B .= B[:,:,idx]
   param[:DCore] .= param[:DCore][idx]
   param[:kAnis] .= param[:kAnis][idx]
  end

  B, param
end

