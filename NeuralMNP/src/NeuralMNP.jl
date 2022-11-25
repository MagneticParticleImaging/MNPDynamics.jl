module NeuralMNP
using LinearAlgebra
using MNPDynamics

using HDF5, BSON
using Flux, NeuralOperators, MLUtils
using Flux: withgradient
using StatsBase #, Statistics
using Random
using ProgressMeter

using Plots

using ImageFiltering

struct NeuralNetworkMNPAlg <: MNPAlgorithm end
const NeuralNetworkMNP = NeuralNetworkMNPAlg()

export NeuralNetworkMNP

@enum FieldType begin
  RANDOM_FIELD
  HARMONIC_RANDOM_FIELD
  HARMONIC_MPI_FIELD
end

export RANDOM_FIELD, HARMONIC_RANDOM_FIELD, HARMONIC_MPI_FIELD

include("magneticFields.jl")
include("neuralOperator.jl")

end