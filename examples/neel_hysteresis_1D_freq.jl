using MNPDynamics
using Plots, Plots.Measures, StaticArrays



samplingMultiplier = 2                  # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*100        # length of time vector
fb = 2.5e6
fx = fb / 100
tMax = (100-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0,stop=tMax,length=tLength);


# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 22e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 5000*[1;0;0]  # anisotropy constant and anisotropy axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = BROWN     # relaxation mode
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 1 / 2.5e6   # warmup time

amp = 0.02
freqs = [1000, 5000, 25e3, 50e3, 100e3]

# No Relaxation

B =  t -> SVector{3,Float64}(2*amp*[cospi(2*fx*t), 0, 0])
pNoRelax = copy(p)
pNoRelax[:relaxation] = NO_RELAXATION
m = simulationMNP(B, t; pNoRelax...)
BNoRelax = getindex.(B.(t),1)
mNoRelax = m[:,1]

# With Brown Relaxation 

msBrown = []
BsBrown = []

for l=1:length(amps)
  global fb = freqs[l]*100
  global fx = fb / 100
  global tMax = (100-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
  global t = range(0,stop=tMax,length=tLength);

  global B =  t -> SVector{3,Float64}(amp*[cospi(2*fx*t), 0, 0])
  global m = simulationMNP(B, t; p...)

  push!(BsBrown, getindex.(B.(t),1)) 
  push!(msBrown, m[:,1])

end

# With Neel Relaxation 

msNeel = []
BsNeel = []

p[:relaxation] = NEEL

for l=1:length(amps)
  global fb = freqs[l]*100
  global fx = fb / 100
  global tMax = (100-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
  global t = range(0,stop=tMax,length=tLength);

  global B =  t -> SVector{3,Float64}(amp*[cospi(2*fx*t), 0, 0])
  global m = simulationMNP(B, t; p...)

  push!(BsNeel, getindex.(B.(t),1)) 
  push!(msNeel, m[:,1])

end

c = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05), RGB(0.5,0.5,0.5)]

p1 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops Brown Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / ???", xlabel="H / T", bottom_margin=5mm, label="No Relax")

p2 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops Neel Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / ???", xlabel="H / T", bottom_margin=5mm, label="No Relax")

for l=1:length(freqs)
  plot!(p1, BsBrown[l],msBrown[l], lw=2, c=c[l], label = "Freq = $(freqs[l])")
  plot!(p2, BsNeel[l],msNeel[l], lw=2, c=c[l], label = "Freq = $(freqs[l])")
end

plot(p1, p2, layout=(2,1), right_margin=5mm, size=(1000,600))