module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using ImageFiltering
using Interpolations
using OffsetArrays

using Distributed
using SharedArrays
using StaticArrays

using ProgressMeter

using HDF5


## sorting algorithms ##

abstract type MNPAlgorithm end

struct FokkerPlanckAlg <: MNPAlgorithm end
struct LangevinFunctionAlg <: MNPAlgorithm end

const FokkerPlanck = FokkerPlanckAlg()
const LangevinFunction = LangevinFunctionAlg()

export MNPAlgorithm, FokkerPlanck, LangevinFunction

@enum RelaxationType begin
  NEEL
  BROWN
  NO_RELAXATION
end

export NEEL, BROWN, NO_RELAXATION

include("utils.jl")
include("sparseMatrixSetup.jl")
include("simulation.jl")
include("multiParams.jl")

export simulationMNP, simulationMNPMultiParams, loadSimulationMNPMultiParams


end # module
