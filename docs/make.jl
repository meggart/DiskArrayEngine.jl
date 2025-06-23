using DiskArrayEngine
using Documenter, DocumenterVitepress

makedocs(;
    modules=[DiskArrayEngine],
    authors="Fabian Gans",
    repo="https://github.com/meggart/DiskArrayEngine.jl",
    sitename="DiskArrayEngine.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/meggart/DiskArrayEngine.jl",
        devurl = "dev",
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/meggart/DiskArrayEngine.jl", # this must be the full URL!
    devbranch = "main",
    push_preview = true,
)