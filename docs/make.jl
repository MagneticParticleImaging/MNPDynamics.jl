using Documenter, MNPDynamics

makedocs(;
    #doctest = true,
    #strict = :doctest,
    modules = [MNPDynamics],
    checkdocs = :exports,
    sitename = " ",
    authors = "Tobias Knopp and contributors",
    repo="https://github.com/MagneticParticleImaging/MNPDynamics.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MagneticParticleImaging.github.io/MNPDynamics.jl",
        assets=String[],
    ),
    pages = [
      "Home" => "index.md",
      "Overview" => "overview.md",
      "API" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/MagneticParticleImaging/MNPDynamics.jl.git",
)

