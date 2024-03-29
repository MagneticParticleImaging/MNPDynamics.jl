
export generateRandomFields, generateRandomParticleParams, generateSnippets

function generateRandomFields(t, Z; fieldType::FieldType, 
                                             dims = 1:3,
                                             freqInterval = (20e3, 50e3),
                                             frequencies = nothing,
                                             filterFactor = (4,20),
                                             numFrequencies = (1,1),
                                             maxField = 30e-3 )

  if fieldType == RANDOM_FIELD
    B = rand_interval(-1, 1, length(t), 3, Z)
    for z=1:Z
      for d=1:3
        if d ∈ dims
          filtFactor = rand_interval(filterFactor[1], filterFactor[2])
          B[:,d,z] = imfilter(B[:,d,z], Kernel.gaussian((filtFactor,))) 
          B[:,d,z] ./= maximum(abs.(B[:,d,z]))
          B[:,d,z] .= maxField*(rand()*B[:,d,z])
        else
            B[:,d,z] .= 0
        end
      end
    end
  elseif fieldType == BANDPASS_RANDOM_FIELD
    B = rand_interval(-1, 1, length(t), 3, Z)
    samplingRate = 1/(t[2]-t[1])
    for z=1:Z
      for d=1:3
        if d ∈ dims
          filter = bandPassFilter(length(t)÷2+1, samplingRate, freqInterval)
          B[:,d,z] =  irfft( rfft(B[:,d,z], 1) .* filter, length(t), 1) 
          B[:,d,z] ./= maximum(abs.(B[:,d,z]))
          B[:,d,z] .= maxField*(rand()*B[:,d,z])
        else
          B[:,d,z] .= 0
        end
      end
    end
  elseif fieldType == RANDOM_SPARSE_FREQUENCY_FIELD
    B = zeros(Float32, length(t), 3, Z)
    numFreq = rand(numFrequencies[1]:numFrequencies[2])
    for z=1:Z
      for d in dims
        for k=1:numFreq
          f = rand_interval(freqInterval[1], freqInterval[2])
          phase = rand_interval(-π,π)
          B[:,d,z] .+= rand()*sin.(2*π*f*t.+phase)
        end
        B[:,d,z] ./= maximum(abs.(B[:,d,z]))
        γ = rand()
        offset = rand_interval(-1,1)
        B[:,d,z] .= (γ*B[:,d,z] .+ (1-γ)*offset)
        B[:,d,z] .= maxField*(rand()*B[:,d,z])
      end
    end
  elseif fieldType == HARMONIC_MPI_FIELD
    B = zeros(Float32, length(t), 3, Z)
    freq = frequencies
    for z=1:Z
      for d in dims
        B[:,d,z] = maxField*rand()*(sin.(2*pi*freq[d]*t) .+ (1-γ)*offset)
      end
    end    
  else
    error("field type $fieldType not supported!")
  end

  return B
end

function generateRandomParticleParams(params, Z; anisotropyAxis=nothing,
                                              boxing = true, γ=1.0)
  paramsInner = copy(params)

  if haskey(params, :DCore) && typeof(params[:DCore]) <: Tuple &&
     haskey(params, :kAnis) && typeof(params[:kAnis]) <: Tuple && boxing

    DCore = zeros(Z)
    KCore = zeros(Z)

    N = 2 
    z = 1

    while z <= Z
      
      DGrid = range(0, 1, length=N+1).^(γ).*(params[:DCore][2] - params[:DCore][1]).+params[:DCore][1]
      KGrid = range(0, 1, length=N+1).^(γ).*(params[:kAnis][2] - params[:kAnis][1]).+params[:kAnis][1]
      indices = CartesianIndices((N,N))
      randidx = shuffle!(collect(1:N*N))
      for q = 1:length(randidx)
        idx = indices[randidx[q]]
        ΔDGrid =  DGrid[idx[1]+1] - DGrid[idx[1]]
        ΔKGrid =  KGrid[idx[2]+1] - KGrid[idx[2]]
        DCore[z] = rand_interval(DGrid[idx[1]], DGrid[idx[1]+1], 1)[1]
        KCore[z] = rand_interval(KGrid[idx[2]], KGrid[idx[2]+1], 1)[1]
        z += 1
        if z > Z
          break
        end
      end
      N *= 2
    end

    paramsInner[:DCore] = DCore
    if anisotropyAxis == nothing
      paramsInner[:kAnis] = [ KCore[z]*randAxis() for z=1:Z ]
    else
      paramsInner[:kAnis] = [ KCore[z]*anisotropyAxis for z=1:Z ]
    end
  else
    if haskey(params, :DCore) && typeof(params[:DCore]) <: Tuple
      DGrid = rand_interval(0, 1, Z).^(γ).*(params[:DCore][2] - params[:DCore][1]).+params[:DCore][1]
      paramsInner[:DCore] = DGrid
    end

    if haskey(params, :kAnis) && typeof(params[:kAnis]) <: Tuple
      KGrid = rand_interval(0, 1, Z).^(γ).*(params[:kAnis][2] - params[:kAnis][1]).+params[:kAnis][1]
      if anisotropyAxis == nothing
        paramsInner[:kAnis] = [ KGrid[z]*randAxis() for z=1:Z ]
      else
        paramsInner[:kAnis] = [ KCore[z]*anisotropyAxis for z=1:Z ]
      end
    end
  end
  return paramsInner
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

function  bandPassFilter(N, samplingRate, freqInterval)
  # indices of the bandpass
  x0 = Int64( (N*freqInterval[1] )÷samplingRate+1 )
  x1 = Int64( (N*freqInterval[2] )÷samplingRate+1 )
  #filter
  filt = zeros(Int64, N)
  filt[x0:x1] .= 1

  return filt
end


function generateSnippets(Xs, Ys, numData, weights, snippetLength)
  weights = collect(weights)
  weights ./= sum(weights)
  N = length(Xs)
  numDataEachSet = zeros(Int, N)
  counter = numData
  for l = 1:(N-1)
    numDataEachSet[l] = round(Int, numData*weights[l])
    counter -= numDataEachSet[l]
  end
  numDataEachSet[end] = counter

  @assert sum(numDataEachSet) == numData

  XOut = zeros(eltype(Xs[1]), snippetLength, size(Xs[1],2), numData)
  YOut = zeros(eltype(Ys[1]), snippetLength, size(Ys[1],2), numData)

  counter = 1
  for l = 1:N
    M = size(Xs[l],3)
    numSnippetEachConfiguration = ceil(Int, numDataEachSet[l] / M)
    currConfig = 1
    for j = 1:numDataEachSet[l]
      currOffset = max(size(Xs[l],1) - snippetLength + 1, 1)
      if numSnippetEachConfiguration > 1
        currOffset = max(currOffset -
                          floor(Int, (size(Xs[l],1) - snippetLength) * ((j-1)÷M) 
                              / (numSnippetEachConfiguration-1)), 1)

      end
      #if mod1(j,M) == M  # just for debugging purpose
      #  println(currOffset)
      #end

      XOut[:,:,counter] .= Xs[l][currOffset:currOffset+snippetLength-1,:,currConfig]
      YOut[:,:,counter] .= Ys[l][currOffset:currOffset+snippetLength-1,:,currConfig]
      currConfig = mod1(currConfig+1, M)
      counter += 1
    end
  end
  return XOut, YOut
end