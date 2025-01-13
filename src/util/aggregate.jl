using StatsBase: rle
using OnlineStats: OnlineStat
export aggregate_diskarray

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

windows_from_spec(::ReduceAggregator, windowsize::Nothing, sdim) = 1:sdim, fill(1, sdim)

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

windows_from_spec(::DirectAggregator, windowsize::Nothing, sdim) = [1:sdim], 1:1

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
    outspecs = create_outwindows(length.(outwindows), windows=outwindows, ismem=ismem, chunks=outchunks)
    return GMDWop(tuple(inars),tuple(outspecs),agg.f)
end

function aggregate_diskarray(a,f,dimspec;skipmissing=false,strategy=:auto)
    
    hasmissings = Missing <: eltype(a)
    if strategy == :reduce || ((isa(f,DataType) || isa(f,UnionAll)) && f <: OnlineStat)
        agg = ReduceAggregator(disk_onlinestat(f))
        op = gmwop_for_aggregator(agg,dimspec,a)
        results_as_diskarrays(op)[1]
    elseif strategy == :direct
        rett = Base.promote_op(f,Vector{Base.nonmissingtype(eltype(a))})
        if hasmissings
            rett = Union{rett,Missing}
        end
        agg = DirectAggregator(create_userfunction(f,rett))
        op = gmwop_for_aggregator(agg,dimspec,a)
        results_as_diskarrays(op)[1]
    elseif strategy == :auto
        rett = Base.promote_op(f,Vector{Base.nonmissingtype(eltype(a))})
        if hasmissings
            rett = Union{rett,Missing}
        end
        agg1 = DirectAggregator(create_userfunction(f,rett))
        agg2 = ReduceAggregator(disk_onlinestat(f))
    
        op1 = gmwop_for_aggregator(agg1,dimspec,a)
        p1 = optimize_loopranges(op1,5e8)
        op2 = gmwop_for_aggregator(agg2,dimspec,a)
        p2 = optimize_loopranges(op2,5e8)
        c1 = actual_io_costs(p1)
        c2 = actual_io_costs(p2)
        #we still prefer direct aggregatoin, so we giv it a slight lead:
        op = c1*0.9 < c2 ? op1 : op2
        results_as_diskarrays(op)[1]
    else
        throw(ArgumentError("Unknown strategy, choose one of `:auto`,`reduce` or `direct`"))
    end
  end
