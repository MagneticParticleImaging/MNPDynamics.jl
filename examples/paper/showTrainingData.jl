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

filenameTrain = "data/trainData4.h5"

BTrain, pTrain = generateStructuredFields(p, tSnippet, p[:numData]; fieldType=RANDOM_FIELD) 
@time mTrain, BTrain = simulationMNPMultiParams(filenameTrain, BTrain, tSnippet, pTrain)


μ₀ = 4*π*1e-7
fig = Figure( figure_padding=4, resolution = (500, 600), fontsize = 16)
ax1 = Axis(fig[1, 1], title = "Magnetic Field", xlabel=L"t / \textrm{ms}", ylabel=L"H / \textrm{mT}")
ax2 = Axis(fig[2, 1], title = "Magnetic Moment", xlabel=L"t / \textrm{ms}", ylabel=L"m / (\textrm{Am}^{2})")
N = 4

for j=1:N
    CairoMakie.lines!(ax1, tSnippet, BTrain[:,1,j]*1000, 
                      color = CairoMakie.RGBf(colors[j]...), linewidth=2)
end

for j=1:N
  CairoMakie.lines!(ax2, tSnippet, mTrain[:,1,j] *  μ₀,# / μ0 / 1000, 
                    color = CairoMakie.RGBf(colors[j]...), linewidth=2)
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


save("trainingData.pdf", fig)