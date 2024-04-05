struct PartialFunctionChain{F1,F2,ARG1,ARG2}
    func1::F1
    func2::F2
    arg1::Val{ARG1}
    arg2::Val{ARG2}
  end
  function(p::PartialFunctionChain{F1,F2,ARG1,ARG2})(x...) where {F1,F2,ARG1,ARG2}
    arg1 = map(Base.Fix1(getindex,x),ARG1)
    r = p.func1(arg1...)
    arg2 = map(ARG2) do (fromout,i)
      fromout ? r : x[i]
    end
    p.func2(arg2...)
  end

struct DirectMerge end
struct BlockMerge
  possible_breaks::Vector{Int}
end


function merge_operations(::DirectMerge,inconn,outconn,to_eliminate)
    is_multioutput1 = length(inconn.outputids) > 1
    if is_multioutput1
        error("Not implemented")
    end
    if inconn.f.f.m isa Mutating
    error("Can not stack mutating function")
    end
    @assert inconn.f.red === nothing
    @assert inconn.f.f isa ElementFunction
    @assert outconn.f.f isa ElementFunction
    outmutating = outconn.f.f.m isa Mutating
    @assert only(inconn.outputids) == to_eliminate
    arg1 = ntuple(i->i+outmutating,length(inconn.inputids))
    inow = length(arg1)+1
    arg2 = Tuple{Bool,Int}[]
    if outmutating 
    push!(arg2,(false,1))
    end
    for id in outconn.inputids
    if id == to_eliminate
        push!(arg2,(true,1))
    else
        push!(arg2,(false,inow))
        inow=inow+1
    end
    end
    arg2 = (arg2...,)
    newinnerf = PartialFunctionChain(inconn.f.f.f,outconn.f.f.f,Val(arg1),Val(arg2))
    newfunc = ElementFunction(newinnerf,outconn.f.f.m)
    UserOp(
        newfunc,
        outconn.f.red,
        outconn.f.init,
        (inconn.f.filters...,outconn.f.filters...),
        outconn.f.finalize,
        outconn.f.buftype,
        outconn.f.outtype,
        inconn.f.allow_threads && outconn.f.allow_threads,
    )
end

function replace_dimids(lw::LoopWindows,dimidmap)
  lr = getloopinds(lw)
  newlr = map(lr) do l
    if haskey(dimidmap,l)
      dimidmap[l]
    else
      l
    end
  end
  LoopWindows(lw.windows,Val(newlr))
end

function create_loopdimmap(inconn,outconn,i_eliminate)
  i1 = findfirst(==(i_eliminate),inconn.outputids)
  i2 = findfirst(==(i_eliminate),outconn.inputids)
  old_lr_1 = map(getloopinds,inconn.outwindows)
  old_lr_2 = map(getloopinds,outconn.inwindows)
  dimidmap = Dict(Pair.(old_lr_2[i2],old_lr_1[i1]))
  ndimsold = max(
    maximum(i->maximum(getloopinds(i),init=0),inconn.inwindows),
    maximum(i->maximum(getloopinds(i),init=0),inconn.outwindows)
  )
  for i in (outconn.inwindows...,outconn.outwindows...)
    lr = getloopinds(i)
    for il in lr
      if !in(il,keys(dimidmap))
        ndimsold += 1
        dimidmap[il] = ndimsold
      end
    end
  end
  dimidmap  
end

#Determine possible window breaks of the node operation where units are in the domain
# of the downstream window operation
function possible_breaks(dg,inode)
  mynode,mydim = dg.nodes[inode]
  inputnodes = Graphs.inneighbors(dg.nodegraph,mynode)
  outputnodes = Graphs.outneighbors(dg.nodegraph,mynode)
  inwindows = map(inputnodes) do innode
    inedge,iinput,ioutput = get_edge(dg.nodegraph,innode,mynode)
    inedge.outwindows[ioutput].windows.members[mydim]
  end
  outwindows = map(outputnodes) do outnode
    outedge,iinput,ioutput = get_edge(dg.nodegraph,mynode,outnode)
    outedge.inwindows[iinput].windows.members[mydim]
  end
  if isempty(inwindows) || isempty(outwindows) 
    return nothing
  end
  if allequal([inwindows;outwindows])
    return DirectMerge()
  end
  startval = minimum(windowminimum,inwindows)
  maxval = maximum(windowmaximum,outwindows)
  breakvals = Int[]
  endval = startval
  while endval <= maxval
    endval_candidates_in = map(inwindows) do iw
      lastcandidate = last_contains_value(iw,endval)
      if lastcandidate > length(iw)
        return maxval+1
      else
        maximum(iw[lastcandidate])
      end
    end
    endval_candidates_out = map(outwindows) do iw
      lastcandidate = last_contains_value(iw,endval)
      if lastcandidate > length(iw)
        return maxval+1
      else
        maximum(iw[lastcandidate])
      end
    end
    if allequal(endval_candidates_in) && allequal(endval_candidates_out) && 
      first(endval_candidates_in)==first(endval_candidates_out)
        push!(breakvals,first(endval_candidates_in))
        endval = endval + 1
    else
      endval = max(maximum(endval_candidates_in),maximum(endval_candidates_out))
    end
  end
  return BlockMerge(breakvals)
end