using MNPDynamics
using Plots, Measures
using FFTW

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 20e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 1250          # anisotropy constant
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL     # relaxation mode
p[:reltol] = 1e-4         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 0.00005     # warmup time
p[:derivative] = true
p[:solver] = :FBDF        # Use more stable solver

# Excitation frequencies
const fx = 2.5e6 / 102
const fy = 2.5e6 / 96

samplingRate = 2.5e6
tLength = lcm(96,102);             # length of time vector
tMax = lcm(96,102) / samplingRate; # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

const amplitude = 0.012

B = (t, offset) -> (amplitude*[sin(2*pi*fx*t), sin(2*pi*fy*t), 0] .+ offset )

nOffsets = (30, 30, 1)

oversampling = 1.25
offsets = vec([ oversampling*amplitude.*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])
anisotropyAxis = vec([ oversampling*2.0.*((Tuple(x).-0.5)./nOffsets.-0.5)  for x in CartesianIndices(nOffsets) ])

@time smM = simulationMNPMultiParams(B, t, offsets; n=anisotropyAxis, p...)

smMFT = reshape(rfft(smM, 1), :, 3, nOffsets...)

pl = Any[]
for my=1:8
  for mx=1:8
    push!(pl, heatmap(abs.(smMFT[(mx-1)*16+(my-1)*17+1, 1, :, :, 1]), 
            c=:viridis, axis=nothing, colorbar=nothing,
            annotations = (2, 4, Plots.text("$mx,$my", :white,:left)) ))
  end
end
plot(pl..., layout=(8,8), size=(1000,1000))
savefig("systemMatrix.svg")
