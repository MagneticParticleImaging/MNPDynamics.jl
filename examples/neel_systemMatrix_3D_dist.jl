using Distributed
@everywhere begin
  using MNPDynamics
  using SharedArrays
end
using Plots
using ProgressMeter

# Parameters
DCore = 20e-9;         # particle diameter in nm
alpha = 0.1;           # damping coefficient
kAnis = 11000;         # anisotropy constant
N = 20;                # maximum spherical harmonics index to be considered
n = [1;0;0];           # anisotropy axis
relaxation = NEEL
reltol = 1e-3
abstol = 1e-6

# Excitation frequencies
fx = 2.5e6 / 102
fy = 2.5e6 / 96
fz = 2.5e6 / 99

samplingRate = 2.5e6
tLength = lcm(96,99,102);       # length of time vector
tMax = tLength / samplingRate;  # maximum evaluation time in seconds

t = range(0, stop=tMax, length=tLength);

nPos = (25, 25, 25)


function calcSystemMatrix(t, nPos; DCore, kAnis, N, reltol, abstol, relaxation, fx, fy, fz)
  smM = SharedArray{Float64}(tLength, 3, nPos...)

  @sync @showprogress @distributed for x in CartesianIndices(nPos)
    #@info "calc $x of $(nPos)"
    # Magnetic field for simulation 
    let fx=fx, fy=fy, fz=fz, nx=x[1], ny=x[2], nz=x[3], nPos=nPos
      B = t -> (0.012*[sin(2*pi*fx*t); sin(2*pi*fy*t); sin(2*pi*fz*t)] .+ 0.012*4*[(nx-0.5)/nPos[1]-0.5; (ny-0.5)/nPos[2]-0.5; (nz-0.5)/nPos[3]-0.5] );
      t, y = simulationMNP(B, t; n, DCore, kAnis, N, reltol, abstol, relaxation)

      smM[:,:,nx,ny,nz] .= y
    end
    #@info "calc $x of $(nPos) done"
  end
  return smM
end

@time smM = calcSystemMatrix(t, nPos; DCore, kAnis, N, reltol, abstol, relaxation, fx, fy, fz);
