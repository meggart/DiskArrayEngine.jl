struct DaggerRunner
    op
    loopranges
    outars
    threaded::Bool
    workerthreads::Bool
    inbuffers_pure
    cbc
    runfilter
end

DaggerRunner(args...; kwargs...) = error("The Dagger extension for DiskArrayEngine is not loaded. Please activate it by running `import Dagger` in your code.")
