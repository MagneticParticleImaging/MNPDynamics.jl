module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

#import Pardiso
#using Sundials

include("utils.jl")
include("Neel.jl")
include("Brown.jl")

end # module
