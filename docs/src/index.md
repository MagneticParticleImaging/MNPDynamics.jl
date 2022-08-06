# MNPDynamics

*Julia package for the simulation of Magnetic Nanoparticle Dynamics*

## Introduction

This package provides Julia based implementation of a Fokker-Plank based magnetic particle simulation
and allows to model Brownian or Néel rotation. It is based on the Matlab package [MNPDynamics](https://github.com/MagneticParticleImaging/MNPDynamics) and implements the discretization using spherical harmonics.
For details on the mathematical background be refer to this [paper](https://arxiv.org/abs/2010.07772).

## Installation

Start julia and open the package mode by entering `]`. Then enter
```julia
add https://github.com/MagneticParticleImaging/MNPDynamics.jl
```
This will install the packages `MNPDynamics.jl` and all its dependencies. 

## License / Terms of Usage

The source code of this project is licensed under the MIT license. This implies that
you are free to use, share, and adapt it. However, please give appropriate credit
by citing the project. You can do so by citing the publication

H. Albers, T. Kluth, and T. Knopp, Simulating magnetization dynamics of large ensembles of single domain nanoparticles: Numerical study of Brown/Néel dynamics and parameter identification problems in magnetic particle imaging, Journal of Magnetism and Magnetic Materials, 541, 168508, 2022, [link](https://www.sciencedirect.com/science/article/abs/pii/S0304885321007678), [*arXiv:2010.07772*](https://arxiv.org/abs/2010.07772)

A BibTeX file `MNPDynamics.bib` can be found in the root folder of the Github repository.

## Contact

If you have problems using the software, find bugs or have ideas for improvements please use
the [issue tracker](https://github.com/MagneticParticleImaging/MNPDynamics.jl/issues). For general questions please use
the [discussions](https://github.com/MagneticParticleImaging/MNPDynamics.jl/discussions) section on Github.

## Contributors

* [Tobias Knopp](https://www.tuhh.de/ibi/people/tobias-knopp-head-of-institute.html)
* [Hannes Albers](https://github.com/h-albers)
* [Tobias Kluth](https://github.com/tobias-kluth)

A complete list of contributors can be found on the [Github page](https://github.com/MagneticParticleImaging/MNPDynamics.jl/graphs/contributors).