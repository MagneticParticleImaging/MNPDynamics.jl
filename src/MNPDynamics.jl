module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
#using Sundials

include("Neel.jl")
include("Brown.jl")

end # module
