module MNPDynamics

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SparseDiffTools
using LinearSolve

using ImageFiltering
using OffsetArrays

using Distributed
using SharedArrays

using ProgressMeter


## sorting algorithms ##

abstract type MNPAlgorithm end

struct FokkerPlanckAlg <: MNPAlgorithm end
struct LangevinFunctionAlg <: MNPAlgorithm end

const FokkerPlanck = FokkerPlanckAlg()
const LangevinFunction = LangevinFunctionAlg()

export FokkerPlanck, LangevinFunction

@enum RelaxationType begin
  NEEL
  BROWN
  NO_RELAXATION
end

export NEEL, BROWN, NO_RELAXATION

@enum FieldType begin
  RANDOM_FIELD
  HARMONIC_RANDOM_FIELD
  HARMONIC_MPI_FIELD
end

export NEEL, RANDOM_FIELD, HARMONIC_RANDOM_FIELD, HARMONIC_MPI_FIELD

include("utils.jl")
include("sparseMatrixSetup.jl")
include("simulation.jl")
include("multiParams.jl")
include("magneticFields.jl")

export simulationMNP, simulationMNPMultiParams


end # module
