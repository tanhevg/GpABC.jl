using Documenter, GpABC

local_build = "local" in ARGS

my_makedocs() = makedocs(
    modules = [GpABC],
    format = Documenter.HTML(
        prettyurls = !local_build,
        canonical = "https://tanhevg.github.io/GpABC.jl/stable/",
    ),
    sitename = "GpABC",
    authors = "Tara Abdul Hameed, Fei He, Jonathan Ish-Horowitz, Istvan Klein, Michael Stumpf, Evgeny Tankhilevich",
    pages = [
        "Home" => "index.md",
        "Notation" => "notation.md",
        "Package Overview" => [
            "ABC Parameter Inference" => "overview-abc.md",
            "ABC Model Selection" => "overview-ms.md",
            "LNA" => "overview-lna.md",
            "Gaussian Process Regression" => "overview-gp.md",
            "summary_stats.md"
        ],
        "Examples" => [
            "ABC Parameter Inference" => "example-abc.md",
            "ABC Model Selection" => "example-ms.md",
            "LNA" => "example-lna.md",
            "Gaussian Processes" => "example-gp.md",
        ],
        "Reference" => [
            "ABC Basic" => "ref-abc.md",
            "ABC Advanced" => "ref-abc-advanced.md",
            "LNA" => "ref-lna.md",
            "Model Selection" => "ref-ms.md",
            "Gaussian Processes" => "ref-gp.md",
            "Kernels" => "ref-kernels.md"
        ]
    ]
)

my_makedocs()

if !local_build
    deploydocs(
        # deps   = Deps.pip("mkdocs", "python-markdown-math"),
        # deps   = nothing,
        # make   = nothing,
        # target = "build",
        repo   = "github.com/tanhevg/GpABC.jl.git"
    )
end
