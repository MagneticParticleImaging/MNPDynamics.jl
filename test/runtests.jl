using MNPDynamics
using Test, LinearAlgebra, StaticArrays, Aqua

@testset "Aqua" begin
  Aqua.test_all(MPIFiles)
end

include("accuracy.jl")
include("multiParams.jl")
include("neuralMNP.jl")