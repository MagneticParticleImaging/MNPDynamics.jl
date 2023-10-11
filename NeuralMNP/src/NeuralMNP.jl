module NeuralMNP
using LinearAlgebra
using MNPDynamics

using HDF5, BSON
using Flux, NeuralOperators, MLUtils
using Flux: withgradient
using StatsBase #, Statistics
using Random
using ProgressMeter
using Distributions
using FFTW

using TensorBoardLogger, Logging

#using Plots
using CairoMakie

using ImageFiltering

struct NeuralOperatorModel <: AbstractMNPModel end

export NeuralOperatorModel

@enum FieldType begin
  RANDOM_FIELD
  BANDPASS_RANDOM_FIELD
  RANDOM_SPARSE_FREQUENCY_FIELD
  HARMONIC_MPI_FIELD
end

export RANDOM_FIELD, BANDPASS_RANDOM_FIELD, RANDOM_SPARSE_FREQUENCY_FIELD, HARMONIC_MPI_FIELD

include("magneticFields.jl")
include("UNet.jl")
include("neuralOperator.jl")

end