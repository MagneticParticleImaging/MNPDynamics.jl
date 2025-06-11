using MNPDynamics
using Test, LinearAlgebra, StaticArrays, Aqua

@testset "Aqua" begin
  Aqua.test_all(MNPDynamics, ambiguities=false)
end

include("accuracy.jl")
include("multiParams.jl")
include("neuralMNP.jl")