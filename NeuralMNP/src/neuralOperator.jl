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

  return mean( norm_l2(ŷ.-y, dims=[1,2]) ./ norm_l2(y, dims=[1,2]) )
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

function train(model, opt, trainLoader, testLoader, normalization::NormalizationParams; 
               epochs::Integer=10, plotStep=1, plotting=false, device=cpu)

  model = model |> device
  normalization = normalization |> device
  opt_state = Flux.setup(opt, model)
               
  trainLoss = Float32(0.0)

	for epoch in 1:epochs

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

		testLoss = loss(model, testLoader, normalization, device)

    println("epoch=$epoch time=$(t_) trainLoss=$trainLoss  testLoss=$testLoss")

    if epoch%plotStep==0
      test_x, test_y = first(testLoader)
      train_x, train_y = first(trainLoader)

      test_y_true = sqrt.(sum(abs.(back(test_y |> device, normalization)).^2, dims=2)) |> cpu
      test_y_pred = sqrt.(sum(abs.(back(model(test_x|> device), normalization)).^2, dims=2)) |> cpu
      train_y_true = sqrt.(sum(abs.(back(train_y |> device, normalization)).^2, dims=2)) |> cpu
      train_y_pred = sqrt.(sum(abs.(back(model(train_x |> device), normalization)).^2, dims=2)) |> cpu

      p1 = plot(test_y_true[:,1,1], label="true"); plot!(p1,test_y_pred[:,1,1], label="predict", title="test")
      p2 = plot(train_y_true[:,1,1],label="true"); plot!(p2,train_y_pred[:,1,1], label="predict", title="train")
      display(plot(p1, p2, layout=(2,1)))
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

function rand_interval(a, b, num...)
  return rand(num...) .* (b-a) .+ a
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

function applyToArbitrarySignal(neuralNetwork::NeuralNetwork, X; device=cpu, kargs...)
  snippetLength = neuralNetwork.timeLength
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

  m = applyToArbitrarySignal(neuralNetwork, X; kargs...)

  ε = (t[2]-t[1]) / 4

  if haskey(kargs,:derivative) && kargs[:derivative]
    for l=1:length(t)
      BTime[l,:] .= B(t[l]+ε)
    end
  
    X = prepareTestData(neuralNetwork.params, Dict(kargs), t, BTime)
    m2 = applyToArbitrarySignal(neuralNetwork, X; kargs...)
    return (m2.-m) ./ ε 
  else
    return m
  end
end


