import Adapt


### model generator ###

export make_neural_operator_model
function make_neural_operator_model(inputChan, outputChan, modes, width, transform=FourierTransform;
                                    permuted=true)
  return Chain(
    # lift (d + 1)-dimensional vector field to n-dimensional vector field
    # here, d == 1 and n == 64
    permuted ? Conv((1,), inputChan=>width) : Dense(inputChan, width),
    # map each hidden representation to the next by integral kernel operator
    OperatorKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorKernel(width=>width, (modes, ), transform, permuted=permuted),
    # project back to the scalar field of interest space
    permuted ? Conv((1,), width=>128, gelu) : Dense(width, 128, gelu),
    permuted ? Conv((1,), 128=>outputChan) : Dense(128, outputChan),
  )
end

function make_unet_neural_operator_model(inputChan, outputChan, modes, width, transform=FourierTransform;
                                    permuted=true)
  return Chain(
    # lift (d + 1)-dimensional vector field to n-dimensional vector field
    # here, d == 1 and n == 64
    permuted ? Conv((1,), inputChan=>width) : Dense(inputChan, width),
    # map each hidden representation to the next by integral kernel operator
    OperatorKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorUNOKernel(width=>width, (modes, ), transform, gelu, permuted=permuted),
    OperatorUNOKernel(width=>width, (modes, ), transform, permuted=permuted),
    # project back to the scalar field of interest space
    permuted ? Conv((1,), width=>128, gelu) : Dense(width, 128, gelu),
    permuted ? Conv((1,), 128=>outputChan) : Dense(128, outputChan),
  )
end

### normalization function ###

struct NormalizationParams{U}
  mean::U
  std::U
end

Adapt.@adapt_structure NormalizationParams

export normalizeData
function normalizeData(X; dims=3)
  xmean = mean(X; dims)
  xstd = std(X; dims)
  return NormalizationParams(xmean, xstd)
end

export trafo
trafo(X, normalization::NormalizationParams) = trafo(X, normalization.mean, normalization.std)

function trafo(X, mean::AbstractArray{T}, std::AbstractArray{T}) where T
  return (X .- mean) ./ (std .+ 1f-12)
end

export back
back(X, normalization::NormalizationParams) = back(X, normalization.mean, normalization.std)

function back(X::AbstractArray{T}, mean::AbstractArray{T}, std::AbstractArray{T}) where T
  return X .* (std .+ 1f-12) .+ mean
end


### loss functions ###

# function loss(model, x::AbstractArray, y::AbstractArray, normalization::NormalizationParams)
#   yPred = back(model(x), normalization)  
#   y = back(y, normalization)

#   r = Float32(0.0)
#   for l=1:size(y,3)
#     r += norm(vec(yPred[:,:,l] .- y[:,:,l])) / norm(vec(y[:,:,l]))
#   end
#   return r / size(y,3)
# end

norm_l2(x::AbstractArray; dims) = sqrt.(sum(abs2, x; dims=dims))

function loss(model, x::AbstractArray, y::AbstractArray, normalization::NormalizationParams)
  ŷ = back(model(x), normalization.mean, normalization.std)  
  y = back(y, normalization.mean, normalization.std)


  #return mean( norm_l2(ŷ.-y, dims=[1,2]) ./ norm_l2(y, dims=[1,2]) )
  return mean( norm_l2(ŷ.-y, dims=[1,2]) ./ maximum(abs.(y), dims=[1,2]) ./ sqrt(size(y,1)*size(y,2)) )
end

function loss(model, loader::DataLoader, normalization::NormalizationParams, device=cpu)
  l = Float32(0.0)
  Q = 1000
  q = 0
  for (x,y) in loader
    l += loss(model, x |> device, y |> device, normalization)
    q +=1
    if q==Q
      break
    end
  end
  return l / q
end

### training function ###

function train(model, opt, trainLoader, testLoaders, normalization::NormalizationParams; 
               epochs::Integer=10, γ::Float32=0f0, stepSize::Integer=0, plotStep=1,
               logging::Bool = false, plotting=false, device=cpu, preloadToDevice=true)

  model = model |> device
  normalization = normalization |> device
  opt_state = Flux.setup(opt, model)

  if preloadToDevice
    trainLoader = trainLoader |> device
    testLoaders = [ t |> device for t in testLoaders]  
  end
               
  trainLoss = Float32(0.0)

  ηCurrent = opt.eta

  
	if logging
		@info "Logging to tensorboard"
		lg = TBLogger("tensorboard_logs/run", min_level=Logging.Info)
		set_step_increment!(lg, 0) # no auto increment
		set_step!(lg, 0)
		Base.global_logger(lg)

		log_text(lg, "training params/epochs", epochs)
    log_text(lg, "training params/stepSize", stepSize)
    log_text(lg, "training params/γ", γ)
	end


	for epoch in 1:epochs
    if logging
  		set_step!(lg, epoch)
    end

		trainLoss = Float32(0.0)

    t_ = @elapsed begin
      # @showprogress "Epoch $epoch" for (x,y) in trainLoader
      for (x,y) in trainLoader

        loss_, gs = Flux.withgradient(model) do m
          loss(m, x |> device, y |> device, normalization)
        end
        trainLoss += loss_
        Flux.Optimise.update!(opt_state, model, gs[1])

      end
    end
		trainLoss /= length(trainLoader)

		testLosses = [loss(model, testLoader, normalization, device) for testLoader in testLoaders]

    println("epoch=$epoch time=$(t_) trainLoss=$trainLoss  testLosses=$(join(string.(testLosses).*" "))")

    if logging
      @info "TrainLoss" loss = trainLoss
      for r = 1:length(testLosses) 
        @info "ValidationLoss $r" loss = testLosses[r]
      end
    end
    if epoch%plotStep==0

      train_x, train_y = first(trainLoader)
      train_y_true = sqrt.(sum(abs.(back(train_y |> device, normalization)).^2, dims=2)) |> cpu
      train_y_pred = sqrt.(sum(abs.(back(model(train_x |> device), normalization)).^2, dims=2)) |> cpu
      p = Any[]
      p_ = plot(train_y_true[:,1,1],label="true"); plot!(p_,train_y_pred[:,1,1], label="predict", title="train")
      push!(p, p_)

      for (i,testLoader) in enumerate(testLoaders)
        test_x, test_y = first(testLoader)
        test_y_true = sqrt.(sum(abs.(back(test_y |> device, normalization)).^2, dims=2)) |> cpu
        test_y_pred = sqrt.(sum(abs.(back(model(test_x|> device), normalization)).^2, dims=2)) |> cpu

        p_ = plot(test_y_true[:,1,1], label="true"); plot!(p_,test_y_pred[:,1,1], label="predict", title="test $i")
        push!(p, p_)
      end
      display(plot(p..., layout=(length(p),1)))
    end

    # adjust learning range
    if stepSize > 0 && epoch%stepSize==0
      ηCurrent *= γ
      Flux.Optimisers.adjust!(opt_state, ηCurrent)
    end
	end
  return model |> cpu
end


export NeuralNetwork
struct NeuralNetwork{U,G1,G2}
  model::U
  normalizationX::NormalizationParams{G1}
  normalizationY::NormalizationParams{G2}
  params::Dict{Symbol, Any}
  timeLength::Int
end


### data preparation ###

function rand_interval(a, b, num...; distribution = :uniform)
  if distribution == :uniform
    return rand(num...) .* (b-a) .+ a
  elseif distribution == :chi
    dist = Distributions.Chi(1)
    return clamp.((1.0 .- rand(dist, num...)./3 ) .* (b-a) .+ a, a, b)
  else
    error("Distribution $(distribution) not implemented yet!")
  end
end

function randAxis()
  n = rand_interval(-1,1,3)
  return n / norm(n) # * rand() # in sphere
end



export prepareTrainData
function prepareTrainData(params, t, B)

  Z = size(B, 3)
  useDCore = false
  useKAnis = false
  useTime = params[:trainTimeParam]

  if haskey(params, :DCore) && typeof(params[:DCore]) <: AbstractArray
    useDCore = true
  end

  if haskey(params, :kAnis) && eltype(params[:kAnis]) <: AbstractArray
    useKAnis = true
  end

  numInputs = 3 + useDCore + useKAnis*3 + useTime

  X = zeros(Float32, length(t), numInputs, Z)
  X[:,1:3,:] .= B
  j = 4
  if useDCore
    X[:,j,:] .= kron(ones(length(t)), params[:DCore]')
    j += 1
  end
  if useKAnis


    for z = 1:Z
      Ω = params[:kAnis][z]
      #x_, y_, z_ = params[:kAnis][z]
      #r = sqrt(x_ * x_ + y_ * y_ + z_ * z_)
      #θ = atan(y_, x_)
      #φ = acos(z_ / r)
      #Ω = [r, θ, φ]
      X[:,j:j+2,z] .= kron(ones(length(t)), Ω')
    end



    j += 3
  end
  if useTime
    X[:,j,:] .= kron(range(0,1,length=length(t)), ones(Z)')
    j += 1
  end

  return X
end

function prepareTrainData(params, t, B, m)
  X = prepareTrainData(params, t, B)
  Y = Float32.(m)
  return X, Y
end

function prepareTestData(paramsTrain, paramsTest, t, B)

  useDCore = false
  useKAnis = false
  useTime = paramsTrain[:trainTimeParam]

  if haskey(paramsTrain, :DCore) && typeof(paramsTrain[:DCore]) <: Tuple
    useDCore = true
  end

  if haskey(paramsTrain, :kAnis) && typeof(paramsTrain[:kAnis]) <: Tuple
    useKAnis = true
  end

  numInputs = 3 + useDCore + useKAnis*3 + useTime

  X = zeros(Float32, length(t), numInputs, 1)
  X[:,1:3,:] .= B
  j = 4
  if useDCore
    X[:,j,1] .= paramsTest[:DCore]*ones(length(t))
    j += 1
  end
  if useKAnis
    Ω = collect(paramsTest[:kAnis])
    #x_, y_, z_ = collect(paramsTest[:kAnis])
    #r = sqrt(x_ * x_ + y_ * y_ + z_ * z_)
    #θ = atan(y_, x_)
    #φ = acos(z_ / r)
    #Ω = [r, θ, φ]
    X[:,j:j+2,1] .= kron(ones(length(t)), Ω')
    j += 3
  end
  if useTime
    X[:,j,:] .= range(0,1,length=length(t))
    j += 1
  end

  return X
end



hannWindow(M) = (1.0 .- cos.(2*π/(M-1)*(0:(M-1))))/(M-1)*M .+ 0.001

function applyToArbitrarySignal(neuralNetwork::NeuralNetwork, X, snippetLength; device=cpu, kargs...)
  N = size(X,1)
  stepSize = snippetLength ÷ 2

  win = hannWindow(snippetLength)
  
  # move model to GPU/CPU
  model = neuralNetwork.model |> device
  nX = neuralNetwork.normalizationX |> device
  nY = neuralNetwork.normalizationY |> device

  weights = zeros(Float32, N)
  output = zeros(Float32, N, 3)
  ranges = UnitRange{Int64}[]
  currStart = 1
  while true
    if currStart+snippetLength-1 <= N
      r = currStart:(currStart+snippetLength-1)
      stop = false
    else
      r = (N-snippetLength+1):N
      stop = true
    end
    push!(ranges, r)
    if stop
      break
    end
    currStart += stepSize
  end

  input = zeros(Float32, snippetLength, size(X,2), length(ranges)) 
  for (i,r) in enumerate(ranges)
    input[:,:,i] = trafo(Float32.(X[r,:,1:1]), neuralNetwork.normalizationX)
  end

  out = back(model(input |> device), nY) |> cpu

  for (i,r) in enumerate(ranges)
    output[r,:] .+= out[:,:,i].*win 
    weights[r] += win
  end  
  
  return output ./ weights
end


function MNPDynamics.simulationMNP(B::g, t, ::NeuralNetworkMNPAlg;
                       neuralNetwork::NeuralNetwork,
                       kargs...
                       ) where g

  BTime = zeros(Float32, length(t), 3)
  for l=1:length(t)
    BTime[l,:] .= B(t[l])
  end

  X = prepareTestData(neuralNetwork.params, Dict(kargs), t, BTime)

  snippetLength = neuralNetwork.timeLength*floor(Int, (1/step(t)) / neuralNetwork.params[:samplingRate])

  m = applyToArbitrarySignal(neuralNetwork, X, snippetLength; kargs...)

  ε = (t[2]-t[1]) / 4

  if haskey(kargs,:derivative) && kargs[:derivative]
    for l=1:length(t)
      BTime[l,:] .= B(t[l]+ε)
    end
  
    X = prepareTestData(neuralNetwork.params, Dict(kargs), t, BTime)
    m2 = applyToArbitrarySignal(neuralNetwork, X, snippetLength; kargs...)
    return (m2.-m) ./ ε 
  else
    return m
  end
end