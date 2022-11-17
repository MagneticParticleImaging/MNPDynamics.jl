


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

### normalization function ###

struct NormalizationParams{U}
  mean::U
  std::U
end

export normalizeData
function normalizeData(X; dims=3)
  xmean = mean(X; dims)
  xstd = std(X; dims)
  return NormalizationParams(xmean, xstd)
end

export trafo
function trafo(X, normalization::NormalizationParams, )
  return (X .- normalization.mean) ./ (normalization.std .+ 1e-12)
end

export back
function back(X, normalization::NormalizationParams)
  return X .* (normalization.std .+ 1e-12) .+ normalization.mean
end


### loss functions ###

function loss(model, x::AbstractArray, y::AbstractArray, normalization::NormalizationParams)
  yPred = back(model(x), normalization)  
  y = back(y, normalization)

  r = Float32(0.0)
  for l=1:size(y,3)
    r += norm(vec(yPred[:,:,l] .- y[:,:,l])) / norm(vec(y[:,:,l]))
  end
  return r / size(y,3)
end

function loss(model, loader::DataLoader, normalization::NormalizationParams)
  l = Float32(0.0)
  Q = 1000
  q = 0
  for (x,y) in loader
    l += loss(model, x, y, normalization)
    q +=1
    if q==Q
      break
    end
  end
  return l / q
end

### training function ###

function train(model, opt, trainLoader, testLoader, normalization::NormalizationParams; 
               epochs::Integer = 10, plotting = true)

  trainLoss = Float32(0.0)

  ps = Flux.params(model)	

	for epoch in 1:epochs

		trainLoss = Float32(0.0)
    t_ = @elapsed begin
      @showprogress "Epoch $epoch" for (x,y) in trainLoader

        loss_, gs = Flux.withgradient(ps) do
          loss(model, x, y, normalization)
        end
        trainLoss += loss_
        Flux.Optimise.update!(opt, ps, gs)
      end
    end
		trainLoss /= length(trainLoader)

		testLoss = loss(model, testLoader, normalization)

    println("epoch=$epoch time=$(t_) trainLoss=$trainLoss  testLoss=$testLoss")

    if plotting
      test_x, test_y = first(testLoader)
      train_x, train_y = first(trainLoader)

      test_y_true = sqrt.(sum(abs.(back(test_y, normalization)).^2, dims=2))
      test_y_pred = sqrt.(sum(abs.(back(model(test_x), normalization)).^2, dims=2))
      train_y_true = sqrt.(sum(abs.(back(train_y, normalization)).^2, dims=2))
      train_y_pred = sqrt.(sum(abs.(back(model(train_x), normalization)).^2, dims=2))

      p1 = plot(test_y_true[:,1,1], label="true"); plot!(p1,test_y_pred[:,1,1], label="predict", title="test")
      p2 = plot(train_y_true[:,1,1],label="true"); plot!(p2,train_y_pred[:,1,1], label="predict", title="train")
      display(plot(p1, p2, layout=(2,1)))
    end
	end
	return trainLoss
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
function prepareTrainData(params, t, B; useTime = true)

  Z = size(B, 3)
  useDCore = false
  useKAnis = false

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

function prepareTrainData(params, t, B, m; useTime = true)
  X = prepareTrainData(params, t, B; useTime)
  Y = m
  return X, Y
end

function prepareTestData(paramsTrain, paramsTest, t, B; useTime = true)

  useDCore = false
  useKAnis = false

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

function applyToArbitrarySignal(neuralNetwork::NeuralNetwork, X)
  snippetLength = neuralNetwork.timeLength
  N = size(X,1)
  numPatches = ceil(Int, N/snippetLength)
  stepSize = snippetLength ÷ 2

  win = hannWindow(snippetLength) 

  weights = zeros(Float32, N)
  output = zeros(Float32, N, 3)
  currStart = 1
  while true
    if currStart+snippetLength-1 <= N
      r = currStart:(currStart+snippetLength-1)
      stop = false
    else
      r = (N-snippetLength+1):N
      stop = true
    end
    xc = X[r,:,1:1]
    xc = trafo(xc, neuralNetwork.normalizationX)
    yc = back(neuralNetwork.model(Float32.(xc)), neuralNetwork.normalizationY)
    output[r,:] .+= yc[:,:,1].*win 
    weights[r] += win
    currStart += stepSize

    if stop
      break
    end
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

  X = prepareTestData(neuralNetwork.params, Dict(kargs), t, BTime; useTime = true)
  m = applyToArbitrarySignal(neuralNetwork, X)

  ε = (t[2]-t[1]) / 4

  if haskey(kargs,:derivative) && kargs[:derivative]
    for l=1:length(t)
      BTime[l,:] .= B(t[l]+ε)
    end
  
    X = prepareTestData(neuralNetwork.params, Dict(kargs), t, BTime; useTime = true)
    m2 = applyToArbitrarySignal(neuralNetwork, X)
    return (m2.-m) ./ ε 
  else
    return m
  end
end


