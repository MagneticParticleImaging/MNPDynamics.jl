using MNPDynamics, NeuralMNP
using Serialization
using Plots, StatsPlots
using Flux, NeuralOperators

filenameModel = "model.bin"
NOModel = deserialize(filenameModel)
include("params.jl")

function plotExampleSignals(model, kAnis=1100, offset=[0,0,0])
  
  p = Dict{Symbol,Any}()
  p[:α] = 0.1               # damping coefficient
  p[:kAnis] = kAnis*[1;0;0]  # anisotropy constant and anisotropy axis
  p[:N] = 20                # maximum spherical harmonics index to be considered
  p[:relaxation] = NEEL     # relaxation mode
  p[:reltol] = 1e-6         # relative tolerance
  p[:abstol] = 1e-6         # absolute tolerance
  p[:tWarmup] = 0.00005     # warmup time
  p[:derivative] = true   
  
  amplitude = 0.012
  fx = 25000;
  tMax = 2/fx; 
  tLength = round(Int, tMax*model.params[:samplingRate]);  

  t = range(0,step=1/model.params[:samplingRate],length=tLength);

  DCore = collect(18:2:24)

  pl1 = plot()

  for d=1:length(DCore)
    p[:DCore] = DCore[d]*1e-9         # particle diameter in nm

    # Magnetic field for simulation 
    B = t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ offset);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:model] = NeuralOperatorModel()
    yNO = simulationMNP(B, t; pNO...)

    plot!(pl1, t[:], y[:,1], lw=2, c=d, label="D=$(DCore[d]) nm true", legend = :outertopright)
    plot!(pl1, t[:], yNO[:,1], lw=2, ls=:dot, c=d, label="D=$(DCore[d]) nm predict")
  end

  kAnis = collect([0, 500, 1200, 1550])

  pl2 = plot()
  p[:DCore] = 20.0e-9         # particle diameter in nm
  for k=1:length(kAnis)
    p[:kAnis] = kAnis[k]*[1.0;0.0;0]  # anisotropy constant and anisotropy axis

    # Magnetic field for simulation 
    B =  t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ offset);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:model] = NeuralOperatorModel()
    yNO = simulationMNP(B, t; pNO...)

    plot!(pl2, t[:], y[:,1], lw=2, c=k, label="kAnis=$(kAnis[k])  true", legend = :outertopright)
    plot!(pl2, t[:], yNO[:,1], lw=2, ls=:dot, c=k, label="kAnis=$(kAnis[k])  predict")
  end


  pl3 = plot()
  p[:DCore] = 20e-9         # particle diameter in nm
  p[:kAnis] = 1000*[1;0;0]
  off = collect([-12, -6, 6, 12])

  for k=1:length(off)

    # Magnetic field for simulation 
    B =  t -> (amplitude*[-cos(2*pi*fx*t); 0*t; 0*t] .+ [off[k]*1e-3, 0.0, 0.0]);

    @time y = simulationMNP(B, t; p...)
    pNO = copy(p)
    pNO[:neuralNetwork] = model
    pNO[:model] = NeuralOperatorModel()
    yNO = simulationMNP(B, t; pNO...)

    plot!(pl3, t[:], y[:,1], lw=2, c=k, label="off=$(off[k]) mT  true", legend = :outertopright)
    plot!(pl3, t[:], yNO[:,1], lw=2, ls=:dot, c=k, label="off=$(off[k]) mT  predict")
  end

  p_ = plot(pl1, pl2, pl3, layout=(3,1), size=(800,600))
  savefig(p_, "evalMPS.pdf")
  p_
end
plotExampleSignals(NOModel, 700, [0.0,0,0])