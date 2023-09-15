using StatsBase: rle

struct DirectAggregator{F<:UserOp} 
    f::F
end
struct ReduceAggregator{F<:UserOp}
    f::F
end


function outrepfromrle(nts)
    r = Int[]
    for i in 1:length(nts)
        for _ in 1:nts[i]
            push!(r,i)
        end
    end
    r
end

function windows_from_spec(::ReduceAggregator,groups,sdim)
    length(groups) == sdim || throw(ArgumentError("Length of group vector must be equal to size of the respective dimension"))
    groupid,n = rle(groups)
    allunique(groupid) || throw(ArgumentError("Aggregation to cyclic groups not yet implemented"))
    1:sdim,  outrepfromrle(n)
end

function windows_from_spec(::ReduceAggregator,windowsize::Int,sdim)    
    1:sdim,[((i-1)÷windowsize)+1 for i in 1:sdim]
end

windows_from_spec(::ReduceAggregator,windowsize::Nothing,sdim) = 1:sdim,nothing

function windows_from_spec(::DirectAggregator,groups,sdim)
    groupid,n = rle(groups)
    allunique(groupid) || throw(ArgumentError("Aggregation to cyclic groups not yet implemented"))
    inwindow = if allequal(n)
        MovingWindow(1,first(n),first(n),length(n))
    else
        cums = [0;cumsum(n)]
        stepvectime = [cums[i]+1:cums[i+1] for i in 1:length(cums)-1]
        to_window(stepvectime)
    end
    outwindow = 1:length(inwindow)
    inwindow,outwindow
end

function windows_from_spec(::DirectAggregator,windowsize::Int,sdim)    
    MovingWindow(1,windowsize,windowsize,ceil(Int,sdim/windowsize)),1:ceil(Int,sdim/windowsize)
end

windows_from_spec(::DirectAggregator,windowsize::Nothing,sdim) = [1:sdim],nothing

function gmwop_for_aggregator(agg,dimspec,inar;ismem=false,outchunks=nothing)
    input_size = size(inar)
    windows = ntuple(ndims(inar)) do idim
        s = input_size[idim]
        ispec = findfirst(i->first(i)==idim,dimspec)
        if ispec !== nothing
            spec = last(dimspec[ispec])
            windows_from_spec(agg,spec,s)
        else
            1:s,1:s
        end
    end
    inwindows = first.(windows)
    outwindows = last.(windows)
    inars = InputArray(inar,windows=(inwindows...,))
    outspecs = if any(isnothing,outwindows)
        ow = filter(!isnothing,outwindows)
        dimmap = findall(!isnothing,outwindows)
        if outchunks === nothing
            outchunks = ntuple(_->nothing,length(dimmap))
        end
        create_outwindows(length.(ow),dimsmap = dimmap, windows=(ow...,),ismem=ismem,chunks=outchunks)
    else
        create_outwindows(length.(outwindows),windows=outwindows,ismem=ismem,chunks=outchunks)
    end
    return GMDWop(tuple(inars),tuple(outspecs),agg.f)
end

