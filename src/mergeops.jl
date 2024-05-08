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


function merge_operations(::Type{<:DirectMerge},inconn,outconn,to_eliminate,dimmap)
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
        outconn.f.finalize,
        outconn.f.buftype,
        outconn.f.outtype,
        inconn.f.allow_threads && outconn.f.allow_threads,
    )
end

function replace_dimids(lw::LoopWindows,dimidmap)
  lr = getloopinds(lw)
  newlr = map(dimidmap,lr)
  LoopWindows(lw.windows,Val(newlr))
end

struct DimMap
  d::Dict{Int,Int}
end
(d::DimMap)(i::Int) = get(d.d,i,i)
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
  DimMap(dimidmap)
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

function merge_operations(::Type{<:BlockMerge},inconn,outconn,to_eliminate,dimmap)
  chain1 = BlockFunctionChain(inconn)
  chain2 = BlockFunctionChain(outconn)
  ifrom = findfirst(==(to_eliminate),inconn.outputids)
  ito = findfirst(==(to_eliminate),outconn.inputids)
  transfer = ifrom => ito

  newfunc = build_chain(chain1,chain2,dimmap,transfer)
  UserOp(
    newfunc,
    outconn.f.red,
    (inconn.f.init...,outconn.f.init...),
    outconn.f.finalize,
    (inconn.f.buftype...,outconn.f.buftype...),
    (inconn.f.outtype...,outconn.f.outtype...),
    inconn.f.allow_threads && outconn.f.allow_threads,
  )
end

struct BlockFunctionChain
  funcs
  args
  transfers::Vector{Pair{Int,Int}} # List of pairs containing the indices of outbuffer-inbuffer pairs to copy data
end
#Constructor to build a length-1 chain
function BlockFunctionChain(f::Union{ElementFunction,BlockFunction},info)
  nin,nout = info
  funcs = [f]
  args = [(ntuple(identity,nin),ntuple(identity,nout))]
  transfers = []
  BlockFunctionChain(funcs,args,transfers)
end
BlockFunctionChain(op::UserOp,info) = BlockFunctionChain(op.f,info)
BlockFunctionChain(c::BlockFunctionChain,_,_,_) = c
describe(c::BlockFunctionChain) = join(map(f->f.f.f),c.funcs)
function build_chain(c1::BlockFunctionChain, c2::BlockFunctionChain, dimmap, transfer)
  argspre = maximum(i->maximum.(i),c1.args)
  args = [c1.args;map(i->map(j->j.+argspre,i),c2.args)]
  funcs = [c1.funcs;c2.funcs]
  inargspre = first(argspre)
  outargspre = last(argspre)
  transfers = copy(c1.transfers)
  for t in c2.transfers
    push!(transfers,(first(t)+outargspre) => (last(t)+inargspre))
  end
  push!(transfers,(first(transfer)+outargspre) => (last(transfer)+inargspre))
  BlockFunctionChain(funcs,args,transfers)
end
maxlr(lw::LoopWindows) = maximum(getloopinds(lw),init=0)
function BlockFunctionChain(conn::MwopConnection)
  nlr = max(maximum(maxlr,conn.inwindows,init=0),maximum(maxlr,conn.outwindows,init=0))
  BlockFunctionChain(conn.f,(length(conn.inwindows),length(conn.outwindows),nlr))
end

struct NestedWindow{P<:AbstractVector} <: AbstractVector{UnitRange{Int}}
  parent::P
  groups::Vector{UnitRange{Int}}
end
Base.size(w::NestedWindow) = (length(w.groups),)
Base.getindex(w::NestedWindow, i::Int) = w.groups[i]
inner_index(g::NestedWindow, i) = collect(g.parent[ip] for j in i for ip in g.groups[j])
inner_range(g::NestedWindow, i) = first(g.groups[first(i)]):last(g.groups[last(i)])
inner_range(g::Window, i) = inner_range(g.w,i)
inner_getindex(w::NestedWindow, i::Int) = w.parent[i]
purify_window(w::NestedWindow) = NestedWindow(purify_window(w.parent),w.groups)

function run_block(f::BlockFunctionChain,inow,inbuffers_wrapped,outbuffers_now,threaded)
  
  func1 = first(f.funcs)
  args1 = first(f.args)
  inbuffers1 = map(Base.Fix1(getindex,inbuffers_wrapped),first(args1))
  outbuffers1 = map(Base.Fix1(getindex,outbuffers_now),last(args1))

  ref_inow = inow


  for buffer in inbuffers1
    foreach(buffer.lw.windows.members,getloopinds(buffer)) do window,li
      if isa(window,NestedWindow) || (isa(window,Window) && isa(window.w,NestedWindow))
        inow = Base.setindex(inow,inner_range(window,inow[li]),li)
      end
    end
  end

  @show ref_inow
  @show inow
  @show typeof(inbuffers1[1].lw.windows)

  run_block(func1,inow,inbuffers1,outbuffers1,threaded)

  @assert length(f.funcs) == length(f.args) == length(f.transfers)+1

  for i in 2:length(f.funcs)
    transfer = f.transfers[i-1]
    for t in transfer
      ob = outbuffers_now[first(t)]
      ib = inbuffers_wrapped[last(t)]
      broadcast!(ob.finalize,ib.a,ob.a)
    end

    func = f.funcs[i]
    args = f.args[i]
    inbuffers = map(Base.Fix1(getindex,inbuffers_wrapped),first(args))
    outbuffers = map(Base.Fix1(getindex,outbuffers_now),last(args))
    inow = ref_inow
    for buffer in inbuffers
      foreach(buffer.lw.windows.members,getloopinds(buffer)) do window,li
        if isa(window,NestedWindow) || (isa(window,Window) && isa(window.w,NestedWindow))
          inow = Base.setindex(inow,inner_range(window,inow[li]),li)
        end
      end
    end
    run_block(func,inow,inbuffers,outbuffers,threaded)

  end

end

function get_groups(windows, strategies)
  Dict(filter(!isnothing,map(windows.windows.members,strategies,getloopinds(windows)) do w,strat,il
    r = blockmerge_groups(w,strat)
    if r === nothing
      nothing
    else
      il=>r
    end
  end))
end

function blockwindows_in(inconn,strategies,i_eliminate)
  outnodetorep = findfirst(==(i_eliminate),inconn.outputids)
  outwindows = inconn.outwindows[outnodetorep]
  groups = get_groups(outwindows,strategies)
  newinwindows = _blockwindows.(inconn.inwindows,(groups,))
  newoutwindows = _blockwindows.(inconn.outwindows,(groups,))
  newinwindows, newoutwindows
end

function blockwindows_out(outconn,strategies,i_eliminate)
  nodetorep = findfirst(==(i_eliminate),outconn.inputids)
  windows = outconn.inwindows[nodetorep]
  groups = get_groups(windows,strategies)
  newinwindows = _blockwindows.(outconn.inwindows,(groups,))
  newoutwindows = _blockwindows.(outconn.outwindows,(groups,))
  newinwindows, newoutwindows
end

function _blockwindows(windows,groups)
  #Now fix the corresponding input windows
  newwindows = map(windows.windows.members,getloopinds(windows)) do w,il
    if haskey(groups,il)
      to_window(NestedWindow(w,groups[il]))
    else
      w
    end
  end
  LoopWindows(ProductArray(newwindows),windows.lr)
end

blockmerge_groups(parent, _) = nothing
function blockmerge_groups(parent, mergestrat::BlockMerge)
  i1 = 1
  groups = UnitRange{Int}[]
  for b in mergestrat.possible_breaks
    inext = findlast(<=(b),parent)
    push!(groups,i1:inext)
    i1 = inext + 1
  end
  groups
end

function merged_connection(::Type{DirectMerge}, _, inconn,outconn,i_eliminate, newop, strategy, dimidmap)

  newinputids = [inconn.inputids;filter(!=(i_eliminate),outconn.inputids)]
  inwindows2 = deepcopy(outconn.inwindows)

  i_keep = findall(!=(i_eliminate),outconn.inputids)
  addinwindows = replace_dimids.(inwindows2[i_keep],(dimidmap,))
  newinwindows = (inconn.inwindows...,addinwindows...)
  newoutwindows = replace_dimids.(outconn.outwindows,(dimidmap,))
  MwopConnection(newinputids,outconn.outputids,newop,newinwindows,newoutwindows)
end

function merged_connection(::Type{BlockMerge}, nodegraph, inconn,outconn,i_eliminate, newop, strategy, dimmap)

  inconninwindows,inconnoutwindows = blockwindows_in(inconn,strategy,i_eliminate)
  outconninwindows,outconnoutwindows = blockwindows_out(outconn,strategy,i_eliminate)
  
  outconninwindows = replace_dimids.(outconninwindows,(dimmap,))
  outconnoutwindows = replace_dimids.(outconnoutwindows,(dimmap,))
  
  outinputids = deepcopy(outconn.inputids)
  ito = findfirst(==(i_eliminate),outinputids)
  outinputids[ito] = length(nodegraph.nodes)+1
  newinputids = [inconn.inputids;outinputids]
  newoutputids = [inconn.outputids;outconn.outputids]
  
  newinwindows = (inconninwindows...,outconninwindows...)
  newoutwindows = (inconnoutwindows..., outconnoutwindows...)
  
  push!(nodegraph.nodes,EmptyInput(nodegraph.nodes[i_eliminate]))

  MwopConnection(newinputids,newoutputids,newop,newinwindows,newoutwindows)
end

