using MNPDynamics
using Plots, Plots.Measures, StaticArrays

# Excitation frequencies
fb = 2.5e6
fx = fb / 102

samplingMultiplier = 2                  # sampling rate = samplingMultiplier*fb
tLength = samplingMultiplier*102        # length of time vector
tMax = (102-1/samplingMultiplier) / fb  # maximum evaluation time in seconds
t = range(0,stop=tMax,length=tLength);

# Parameters
p = Dict{Symbol,Any}()
p[:DCore] = 18e-9         # particle diameter in nm
p[:α] = 0.1               # damping coefficient
p[:kAnis] = 10000*[1;0;0]  # anisotropy constant and anisotropy axis
p[:N] = 20                # maximum spherical harmonics index to be considered
p[:relaxation] = NEEL
p[:reltol] = 1e-6         # relative tolerance
p[:abstol] = 1e-6         # absolute tolerance
p[:tWarmup] = 1 / fb      # warmup time

p[:model] = EquilibriumModel()

amps = [0.002, 0.005, 0.01, 0.02, 0.04]

# No Relaxation

B =  t -> SVector{3,Float64}(amps[end]*[cospi(2*fx*t), 0, 0])
pNoRelax = copy(p)
pNoRelax[:relaxation] = NO_RELAXATION
m = simulationMNP(B, t; pNoRelax...)
BNoRelax = getindex.(B.(t),1)
mNoRelax = m[:,1]

# With Brown Relaxation 

msBrown = []
BsBrown = []
p[:model] = EquilibriumAnisModel()

for l=1:length(amps)
  global B =  t -> SVector{3,Float64}(amps[l]*[cospi(2*fx*t), 0, 0])
  global m = simulationMNP(B, t; p...)

  push!(BsBrown, getindex.(B.(t),1)) 
  push!(msBrown, m[:,1])

end

# With Neel Relaxation 

msNeel = []
BsNeel = []

p[:model] = FokkerPlanckModel()


for l=1:length(amps)
  global B =  t -> SVector{3,Float64}(amps[l]*[cospi(2*fx*t), 0, 0])
  global m = simulationMNP(B, t; p...)

  push!(BsNeel, getindex.(B.(t),1)) 
  push!(msNeel, m[:,1])

end

c = [RGB(0.0,0.29,0.57), RGB(0.3,0.5,0.7), RGB(0.94,0.53,0.12), RGB(0.99,0.75,0.05), RGB(0.5,0.5,0.5)]

p1 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops EqAnis2 Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / a.u.", xlabel="H / T", bottom_margin=5mm, label="No Relax")

p2 = plot(BNoRelax,mNoRelax, title="Hysteresis Loops Neel Relaxation", lw=2.5, c=RGB(0.0,1.0,1.0),  
                    ylabel="m / a.u", xlabel="H / T", bottom_margin=5mm, label="No Relax")

for l=1:length(amps)
  plot!(p1, BsBrown[l],msBrown[l], lw=2, c=c[l], label = "Amp = $(amps[l])")
  plot!(p2, BsNeel[l],msNeel[l], lw=2, c=c[l], label = "Amp = $(amps[l])")
end

plot(p1, p2, layout=(2,1), right_margin=5mm, left_margin=5mm, size=(1000,600))