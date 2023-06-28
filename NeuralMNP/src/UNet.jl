# Python implementation
# https://github.com/gegewen/ufno

### ConvAsymPadding Layer ###

struct ConvAsymPadding{T, N}
  c::T
  pad::NTuple{N,Int}
end
function ConvAsymPadding(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
  init = Flux.glorot_uniform,  stride = 1, pad = 0, dilation = 1, bias=true) where N
  length(pad) < 2N || all(i -> length(unique(pad[i:i+1])) == 1, 1:2:2N) == 1&& return Conv(k, ch , σ, init=init, stride=stride, pad=pad, dilation=dilation, bias=bias)

  pad_manual = Tuple(map(i -> abs(pad[i] - pad[i+1]), 1:2:2N))
  pad_auto = Tuple(map(i -> minimum(pad[i:i+1]), 1:2:2N))
  return ConvAsymPadding(Conv(k, ch, σ, init=init, stride=stride, pad=pad_auto, dilation=dilation), pad_manual)
end
function (c::ConvAsymPadding)(x::AbstractArray)
  # Maybe there are faster ways to do this as well...
  padding = similar(x, c.pad..., size(x)[end-1:end]...)
  fill!(padding, 0)
  c.c(cat(x, padding, dims=1:length(c.pad)))
end

Flux.@functor ConvAsymPadding


### Cropping Layer ###

struct Crop{N}
  crop::NTuple{N,Int}
end

function (c::Crop{2})(x::AbstractArray)
  return x[(1+c.crop[1]):(end-c.crop[2]),:,:]
end

function (c::Crop{4})(x::AbstractArray)
  return x[(1+c.crop[1]):(end-c.crop[2]), (1+c.crop[3]):(end-c.crop[4]),:,:]
end

function (c::Crop{6})(x::AbstractArray)
  return x[(1+c.crop[1]):(end-c.crop[2]), (1+c.crop[3]):(end-c.crop[4]), (1+c.crop[5]):(end-c.crop[6]),:,:]
end

Flux.@functor Crop

### BatchNormWrap ###

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)

function _random_normal(shape...)
  return Float32.(rand(Distributions.Normal(0.f0,0.02f0),shape...)) 
end

function BatchNormWrap(out_ch)
  Chain(x->expand_dims(x,3),
  BatchNorm(out_ch),
  x->reshape(x, size(x)[4:end]))
end

### UNetConvBlock ###

UNetConvBlock(in_chs, out_chs; kernel = (3,3,3), pad = 1, stride=1) =
  Chain(Conv(kernel, in_chs=>out_chs, pad = pad, stride=stride; init=_random_normal),
  BatchNormWrap(out_chs),
  x->leakyrelu.(x,0.02f0))


_interp(N::NTuple{1,Int}) = :nearest
_interp(N::NTuple{2,Int}) = :bilinear
_interp(N::NTuple{3,Int}) = :trilinear

function make_model_unet_skip(N::NTuple{D,Int}, inChan_ = 1, outChan_ = 1; 
                    depth = 3, inputResidual=true, baseChan = 4 ) where D
  catChannels(x,y) = cat(x, y, dims=D+1)

  maxDepth = minimum(round.(Int,log2.(N)).-1)
  numLayers = min(maxDepth, depth)
  
  H = zeros(Int, numLayers, D)
  H[1,:] .= N
  for l=1:numLayers-1
    H[l+1,:] .= ceil.(Int, vec(H[l,:])./2)
  end
  needCrop = Int.(isodd.(H))
  
  interp = _interp(N)
  kernel = ntuple(_->3, D)
  scale = ntuple(_->2, D)

  chan = baseChan * 2^numLayers
  currNet = Chain(
    UNetConvBlock(chan, 2*chan, kernel=kernel, pad=1),
    UNetConvBlock(2*chan, chan, kernel=kernel, pad=1)
  )

  for l=1:numLayers
    chan =  baseChan * 2^(numLayers-l)
    inChan = (l==numLayers) ? inChan_ : chan
  
    if sum(needCrop[end-l+1,:]) > 0
      innerNet = Chain(  MaxPool(scale, pad=ntuple(d->needCrop[end-l+1,d], D)),
                        currNet,
                        Upsample(interp; scale),
                        Crop( ntuple(d -> isodd(d) ? needCrop[end-l+1,(d+1)÷2] : 0, 2*D) ) )
    else
      innerNet = Chain(  MaxPool(scale), 
                          currNet,
                          Upsample(interp; scale) ) 
    end

    currNet = Chain(  UNetConvBlock(inChan, 2*chan; kernel, pad=1),
                      UNetConvBlock(2*chan, 2*chan; kernel, pad=1),
                      SkipConnection( innerNet,
                                      catChannels),     
                      UNetConvBlock(4*chan,2*chan; kernel, pad=1),
                      UNetConvBlock(2*chan,chan; kernel, pad=1)
          )
  end

  currNet = Chain(
    currNet,
    Conv(ntuple(_-> 1, D), baseChan=>outChan_, pad = 0, gelu ;init=_random_normal)
  )

  if inputResidual
    return Chain(
        SkipConnection(
          Chain(
            currNet, 
            Conv(ntuple(_-> 1, D), outChan_=>outChan_, identity; pad = 0, init=_random_normal),
            ),
          (x,y) -> x .+ y[CartesianIndices(size(y)[1:end-2]),1:1,:] #mean(y, dims=4) #Add channel one onto the output
        )
      )
  else
    return currNet
  end
end




###### U-FNO ############

struct OperatorUNOKernel{L, C, Q, F} 
  linear::L
  conv::C
  unet::Q
  σ::F
end

function OperatorUNOKernel(ch::Pair{S, S},
                      modes::NTuple{N, S},
                      Transform::Type{<:NeuralOperators.AbstractTransform},
                      σ = identity;
                      permuted = false) where {S <: Integer, N}
  linear = permuted ? Conv(Tuple(ones(Int, length(modes))), ch) :
           Dense(ch.first, ch.second)
  conv = OperatorConv(ch, modes, Transform; permuted = permuted)
  unet = make_model_unet_skip((200,), ch.first, ch.second, 
                                   depth = 1, inputResidual=true, baseChan = 2)

  return OperatorUNOKernel(linear, conv, unet, σ)
end

Flux.@functor OperatorUNOKernel

function Base.show(io::IO, l::OperatorUNOKernel)
  print(io,
        "OperatorUNOKernel(" *
        "$(l.conv.in_channel) => $(l.conv.out_channel), " *
        "$(l.conv.transform.modes), " *
        "$(nameof(typeof(l.conv.transform))), " *
        "σ=$(string(l.σ)), " *
        "permuted=$(ispermuted(l.conv))" *
        ")")
end

function (m::OperatorUNOKernel)(𝐱)
  return m.σ.(m.linear(𝐱) + m.unet(𝐱) + m.conv(𝐱))
end