```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "DiskArrayEngine.jl"
  text: v0.2.3 (pre-alpha!)
  tagline: Tools for defining and running large computations without out-of-memory issues!
  image:
    <!-- src: /logo.png -->
    alt: DiskArrayEngine
  actions:
    - theme: brand
      text: View on Github
      link: https://github.com/meggart/DiskArrayEngine.jl
    - theme: alt
      text: API
      link: /api
features:
  - icon: <img width="64" height="64" src="https://juliadatacubes.github.io/YAXArrays.jl/dev/logo.png" alt="markdown"/>
    title: How to start using DiskArrayEngine.jl?
    details: This package is not intended for direct use by end-users. Instead, YAXArrays.jl is the recommended entry point for utilizing the features provided here. It is a package designed for operating on out-of-core labeled arrays.
    link: https://juliadatacubes.github.io/YAXArrays.jl/v0.6.1/
  - icon: <img width="64" height="64" src="https://juliaparallel.org/Dagger.jl/dev/assets/logo.svg" alt="markdown"/>
    title: Out-of-core and parallel computing!
    details: Run computations represented as directed acyclic graphs (DAGs) efficiently across multiple Julia worker processes and threads using Dagger.jl.
    link: https://github.com/JuliaParallel/Dagger.jl
  - icon: <img width="64" height="64" src="https://img.icons8.com/?size=100&id=64191&format=png&color=000000" alt="markdown"/>
    title: Local runner
    details: Of course, you can always prototype on your local computer. Test multi-threaded and multi-process execution using Distributed.jl before fully committing to Dagger.jl.
    link: https://juliadatacubes.github.io/YAXArrays.jl/v0.6.1/UserGuide/compute.html#Distributed-Computation
---
```
