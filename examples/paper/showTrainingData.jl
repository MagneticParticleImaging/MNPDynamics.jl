using CairoMakie

using MNPDynamics
using NeuralMNP
using LinearAlgebra
using Statistics
using Random
using Flux

colors = [(0/255,73/255,146/255), # UKE blau
(239/255,123/255,5/255),	# Orange (dunkel)
(138/255,189/255,36/255),	# Grün
(178/255,34/255,41/255), # Rot
(170/255,156/255,143/255), 	# Mocca
(87/255,87/255,86/255),	# Schwarz (Schrift)
(255/255,223/255,0/255), # Gelb
(104/255,195/255,205/255),	# "TUHH"
(45/255,198/255,214/255), #  TUHH
(193/255,216/255,237/255)]

include("params.jl")

filenameTrain = "data/trainData1.h5"

lw=1.5

Random.seed!(seed)
BTrain = generateRandomFields(tBaseData, dfDatasets.numData[1]; 
                              fieldType = dfDatasets.fieldType[1], 
                              dims = dfDatasets.fieldDims[1],
                              filterFactor = dfDatasets.filterFactor[1],
                              maxField = dfDatasets.maxField[1])


pTrain = generateRandomParticleParams(p, dfDatasets.numData[1]; 
                  anisotropyAxis = dfDatasets.anisotropyAxis[1],
                  distribution = dfDatasets.samplingDistribution[1])

@time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tBaseData, pTrain; force=false)


μ₀ = 4*π*1e-7
fig = Figure( figure_padding=1, resolution = (800, 900) )

ax1 = CairoMakie.Axis(fig[1, 1], title = "Magnetic Field", ylabel=L"H_x / \textrm{mT}", xticklabelsvisible=false, xticksvisible=false)
ax2 = CairoMakie.Axis(fig[2, 1], ylabel=L"H_y / \textrm{mT}", xticklabelsvisible=false, xticksvisible=false)
ax3 = CairoMakie.Axis(fig[3, 1], ylabel=L"H_z / \textrm{mT}", xticklabelsvisible=false, xticksvisible=false)
ax4 = CairoMakie.Axis(fig[4, 1], title = "Magnetic Moment", ylabel=L"m_x / (\textrm{Am}^{2})", xticklabelsvisible=false, xticksvisible=false)
ax5 = CairoMakie.Axis(fig[5, 1], ylabel=L"m_y / (\textrm{Am}^{2})", xticklabelsvisible=false, xticksvisible=false)
ax6 = CairoMakie.Axis(fig[6, 1], xlabel=L"t / \textrm{ms}", ylabel=L"m_z / (\textrm{Am}^{2})")
N = 4

linkxaxes!(ax1, ax2, ax3, ax4, ax5, ax6)
CairoMakie.xlims!(ax6, 0, maximum(tSnippet))

for j=1:N
  CairoMakie.lines!(ax1, tSnippet, BTrain[:,1,j]*1000, 
                      color = CairoMakie.RGBf(colors[j]...), linewidth=lw)
  CairoMakie.lines!(ax2, tSnippet, BTrain[:,2,j]*1000, 
                      color = CairoMakie.RGBf(colors[j]...), linewidth=lw)
  CairoMakie.lines!(ax3, tSnippet, BTrain[:,3,j]*1000, 
                      color = CairoMakie.RGBf(colors[j]...), linewidth=lw)

  CairoMakie.lines!(ax4, tSnippet, mTrain[:,1,j] *  μ₀,# / μ0 / 1000, 
                    color = CairoMakie.RGBf(colors[j]...), linewidth=lw)
  CairoMakie.lines!(ax5, tSnippet, mTrain[:,2,j] *  μ₀,# / μ0 / 1000, 
                    color = CairoMakie.RGBf(colors[j]...), linewidth=lw)
  CairoMakie.lines!(ax6, tSnippet, mTrain[:,3,j] *  μ₀,# / μ0 / 1000, 
                    color = CairoMakie.RGBf(colors[j]...), linewidth=lw)
end

#lines(ax1, x, y, color = CairoMakie.RGBf(colors[1]...))
#lines(fig[1, 2], x, y, color = :blue)
#lines(fig[2, 1:2], x, y, color = :green)

#fTD, axTD, lTD1 = CairoMakie.lines(timePoints[steps], dataCompressed[:,1], 
#                        figure = (; figure_padding=4, resolution = (1000, 800), fontsize = 11),
#                        axis = (; title = "Time Domain"),
#                        color = CairoMakie.RGBf(colors[1]...),
#                        label = labels_[1])
#for j=2:size(data,2)
#  CairoMakie.lines!(axTD, timePoints[steps],dataCompressed[:,j], 
#                    color = CairoMakie.RGBf(colors[j]...),  #linewidth=3)
#                    label = labels_[j])
#end


save("img/trainingData.pdf", fig)