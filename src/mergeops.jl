struct PartialFunctionChain{F1,F2,ARG1,ARG2}
  func1::F1
  func2::F2
  arg1::Val{ARG1}
  arg2::Val{ARG2}
end
function (p::PartialFunctionChain{F1,F2,ARG1,ARG2})(x...) where {F1,F2,ARG1,ARG2}
  arg1 = map(Base.Fix1(getindex, x), ARG1)
  r = p.func1(arg1...)
  arg2 = map(ARG2) do (fromout, i)
    fromout ? r : x[i]
  end
  p.func2(arg2...)
end

struct DirectMerge end
struct BlockMerge
  possible_breaks::Vector{Int}
end


function merge_operations(::Type{<:DirectMerge}, inconn, outconn, to_eliminate, dimmap)
  is_multioutput1 = length(inconn.outputids) > 1
  if is_multioutput1
    error("Not implemented")
  end
  if inconn.f.f.m isa Mutating
    error("Can not stack mutating function")
  end
  @assert only(inconn.f.red) === nothing
  @assert inconn.f.f isa ElementFunction
  @assert outconn.f.f isa ElementFunction
  outmutating = outconn.f.f.m isa Mutating
  @assert only(inconn.outputids) == to_eliminate
  arg1 = ntuple(i -> i + outmutating, length(inconn.inputids))
  inow = length(arg1) + 1
  arg2 = Tuple{Bool,Int}[]
  if outmutating
    push!(arg2, (false, 1))
  end
  for id in outconn.inputids
    if id == to_eliminate
      push!(arg2, (true, 1))
    else
      push!(arg2, (false, inow))
      inow = inow + 1
    end
  end
  arg2 = (arg2...,)
  newinnerf = PartialFunctionChain(inconn.f.f.f, outconn.f.f.f, Val(arg1), Val(arg2))
  newfunc = ElementFunction(newinnerf, outconn.f.f.m)
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

function replace_dimids(lw::LoopWindows, dimidmap)
  lr = getloopinds(lw)
  newlr = map(dimidmap, lr)
  LoopWindows(lw.windows, Val(newlr))
end

struct DimMap
  d::Dict{Int,Int}
end
(d::DimMap)(i::Int) = get(d.d, i, i)
function create_loopdimmap(inconn, outconn, i_eliminate)
  i1 = findfirst(==(i_eliminate), inconn.outputids)
  i2 = findfirst(==(i_eliminate), outconn.inputids)
  old_lr_1 = map(getloopinds, inconn.outwindows)
  old_lr_2 = map(getloopinds, outconn.inwindows)
  dimidmap = Dict(Pair.(old_lr_2[i2], old_lr_1[i1]))
  ndimsold = max(
    maximum(i -> maximum(getloopinds(i), init=0), inconn.inwindows),
    maximum(i -> maximum(getloopinds(i), init=0), inconn.outwindows)
  )
  for i in (outconn.inwindows..., outconn.outwindows...)
    lr = getloopinds(i)
    for il in lr
      if !in(il, keys(dimidmap))
        ndimsold += 1
        dimidmap[il] = ndimsold
      end
    end
  end
  DimMap(dimidmap)
end



#Determine possible window breaks of the node operation where units are in the domain
# of the downstream window operation
function possible_breaks(dg::DimensionGraph, inode::Int)
  mynode, mydim = dg.nodes[inode]
  inputnodes = Graphs.inneighbors(dg.nodegraph, mynode)
  outputnodes = Graphs.outneighbors(dg.nodegraph, mynode)
  inwindows = map(inputnodes) do innode
    inedge, iinput, ioutput = get_edge(dg.nodegraph, innode, mynode)
    inedge.outwindows[ioutput].windows.members[mydim]
  end
  outwindows = map(outputnodes) do outnode
    outedge, iinput, ioutput = get_edge(dg.nodegraph, mynode, outnode)
    outedge.inwindows[iinput].windows.members[mydim]
  end
  possible_breaks(inwindows, outwindows)
end

function possible_breaks(inwindows, outwindows)
  if isempty(inwindows) || isempty(outwindows)
    return nothing
  end
  if allequal([inwindows; outwindows])
    return DirectMerge()
  end
  startval = minimum(windowminimum, inwindows)
  maxval = maximum(windowmaximum, outwindows)
  breakvals = Int[]
  endval = startval
  while endval <= maxval
    endval_candidates_in = map(inwindows) do iw
      lastcandidate = last_contains_value(iw, endval)
      if lastcandidate > length(iw)
        return maxval + 1
      else
        maximum(inner_range(iw[lastcandidate]))
      end
    end
    endval_candidates_out = map(outwindows) do iw
      lastcandidate = last_contains_value(iw, endval)
      if lastcandidate > length(iw)
        return maxval + 1
      else
        maximum(inner_range(iw[lastcandidate]))
      end
    end
    if allequal(endval_candidates_in) && allequal(endval_candidates_out) &&
       first(endval_candidates_in) == first(endval_candidates_out)
      push!(breakvals, first(endval_candidates_in))
      endval = first(endval_candidates_in)+1
    else
      newendval = max(maximum(endval_candidates_in), maximum(endval_candidates_out))
      endval = max(endval + 1, newendval)
    end
  end
  return BlockMerge(breakvals)
end

function merge_operations(::Type{<:BlockMerge}, inconn, outconn, to_eliminate, dimmap)
  chain1 = BlockFunctionChain(inconn)
  chain2 = BlockFunctionChain(outconn)
  ifrom = findfirst(==(to_eliminate), inconn.outputids)
  ito = findall(==(to_eliminate), outconn.inputids)
  transfer = ifrom => ito

  newfunc = build_chain(chain1, chain2, dimmap, transfer)
  UserOp(
    newfunc,
    (inconn.f.red..., outconn.f.red...),
    (inconn.f.init..., outconn.f.init...),
    (inconn.f.finalize..., outconn.f.finalize...),
    (inconn.f.buftype..., outconn.f.buftype...),
    (inconn.f.outtype..., outconn.f.outtype...),
    inconn.f.allow_threads && outconn.f.allow_threads,
  )
end

struct BlockFunctionChain
  funcs
  args
  transfers::Vector{Pair{Int,Vector{Int}}} # List of pairs containing the indices of outbuffer-inbuffer pairs to copy data
end
#Constructor to build a length-1 chain
function BlockFunctionChain(f::Union{ElementFunction,BlockFunction}, info)
  nin, nout = info
  funcs = [f]
  args = [(ntuple(identity, nin), ntuple(identity, nout))]
  transfers = Pair{Int,Vector{Int}}[]
  BlockFunctionChain(funcs, args, transfers)
end
BlockFunctionChain(f::BlockFunctionChain, _) = f
BlockFunctionChain(op::UserOp, info) = BlockFunctionChain(op.f, info)
BlockFunctionChain(c::BlockFunctionChain, _, _, _) = c
describe(c::BlockFunctionChain) = join(map(f -> f.f.f), c.funcs)
function build_chain(c1::BlockFunctionChain, c2::BlockFunctionChain, dimmap, transfer)
  argsprein = maximum(i -> maximum(first(i)), c1.args)
  argspreout = maximum(i -> maximum(last(i)), c1.args)
  c2args = map(c2.args) do (argsin, argsout)
    (argsin .+ argsprein), (argsout .+ argspreout)
  end
  args = [c1.args; c2args]
  funcs = [c1.funcs; c2.funcs]
  transfers = copy(c1.transfers)
  for t in c2.transfers
    push!(transfers, (first(t) + argspreout) => (last(t) .+ argsprein))
  end
  push!(transfers, first(transfer) => (last(transfer) .+ argsprein))
  BlockFunctionChain(funcs, args, transfers)
end
maxlr(lw::LoopWindows) = maximum(getloopinds(lw), init=0)
function BlockFunctionChain(conn::MwopConnection)
  nlr = max(maximum(maxlr, conn.inwindows, init=0), maximum(maxlr, conn.outwindows, init=0))
  BlockFunctionChain(conn.f, (length(conn.inwindows), length(conn.outwindows), nlr))
end

function replace_nested_buffers(inbuffers, outbuffers, inow)
  ref_inow = inow
  for buffer in (inbuffers..., outbuffers...)
    foreach(buffer.lw.windows.members, getloopinds(buffer)) do window, li
      if eltype(window) <: WindowGroup
        wnow = window[ref_inow[li]]
        newwindows = first(first(wnow).g):last(last(wnow).g)
        inow = Base.setindex(inow, newwindows, li)
      end
    end
  end
  inow
end

function run_block(f::BlockFunctionChain, ref_inow, inbuffers_wrapped, outbuffers_now, threaded)

  func = first(f.funcs)
  args = first(f.args)
  inbuffers = map(Base.Fix1(getindex, inbuffers_wrapped), first(args))
  outbuffers = map(Base.Fix1(getindex, outbuffers_now), last(args))

  outbuffers_written = Int[]
  transfers_done = Int[]

  foreach(last(args)) do iout
    push!(outbuffers_written, iout)
  end

  inow = replace_nested_buffers(inbuffers, outbuffers, ref_inow)

  # println("Running block 1: ", inow)
  # println(func.f)
  # println("Number of inputs: ", length(inbuffers), " used buffers: ", first(args))
  # println("Number of outputs: ", length(outbuffers), " used buffers: ", last(args))
  
  run_block(func, inow, inbuffers, outbuffers, threaded)

  @assert length(f.funcs) == length(f.args) == length(f.transfers) + 1

  for i in 2:length(f.funcs)
    
    for t in f.transfers
      ifrom = first(t)
      if ifrom in outbuffers_written && !(ifrom in transfers_done)
        itos = last(t)
        ob = outbuffers_now[ifrom]
        for ito in itos
          ib = inbuffers_wrapped[ito]
          #println("Transferring buffer from ", ifrom, " to ", ito)
          if size(ib.a) != size(ob.a)
            broadcast!(ob.finalize, view(ib.a,Base.OneTo.(size(ob.a))...), ob.a)
          else
            broadcast!(ob.finalize, ib.a, ob.a)
          end
          push!(transfers_done, ifrom)
        end
      end
    end

    func = f.funcs[i]
    args = f.args[i]
    inbuffers = map(Base.Fix1(getindex, inbuffers_wrapped), first(args))
    outbuffers = map(Base.Fix1(getindex, outbuffers_now), last(args))

    foreach(last(args)) do iout
      push!(outbuffers_written, iout)
    end

    inow = replace_nested_buffers(inbuffers, outbuffers, ref_inow)

    # println("Running block $(i): ", inow)
    # println(func.f)
    # println("Number of inputs: ", length(inbuffers), " used buffers: ", first(args))
    # println("Number of outputs: ", length(outbuffers), " used buffers: ", last(args))

    run_block(func, inow, inbuffers, outbuffers, threaded)

  end

end

function get_groups(windows, strategies)
  Dict(filter(!isnothing, map(windows.windows.members, strategies, getloopinds(windows)) do w, strat, il
    r = blockmerge_groups(w, strat)
    if r === nothing
      nothing
    else
      il => r
    end
  end))
end

function blockwindows_inconn(inconn, strategies, i_eliminate)
  outnodetorep = findfirst(==(i_eliminate), inconn.outputids)
  outwindows = inconn.outwindows[outnodetorep]
  groups = get_groups(outwindows, strategies[i_eliminate])
  newinwindows = _blockwindows.(inconn.inwindows, (groups,))
  newoutwindows = _blockwindows.(inconn.outwindows, (groups,))
  newinwindows, newoutwindows
end

function blockwindows_outconn(outconn, strategies, i_eliminate)
  nodetorep = findfirst(==(i_eliminate), outconn.inputids)
  windows = outconn.inwindows[nodetorep]
  groups = get_groups(windows, strategies[i_eliminate])
  newinwindows = _blockwindows.(outconn.inwindows, (groups,))
  newoutwindows = _blockwindows.(outconn.outwindows, (groups,))
  newinwindows, newoutwindows
end

function wrap_groupwindow(window, groups)
  groupwindow = if eltype(window) <: WindowGroup
    parent = first(window).parent
    map(groups) do g
      WindowGroup(parent, first(window[first(g)].g):last(window[last(g)].g))
    end
  else
    map(groups) do g
      WindowGroup(window, g)
    end
  end
  to_window(groupwindow)
end

function _blockwindows(windows, groups)
  #Now fix the corresponding input windows
  newwindows = map(windows.windows.members, getloopinds(windows)) do w, il
    if haskey(groups, il)
      groupwindow = wrap_groupwindow(w, groups[il])
      to_window(groupwindow)
    else
      w
    end
  end
  LoopWindows(ProductArray(newwindows), windows.lr)
end

blockmerge_groups(parent, _) = nothing
function blockmerge_groups(parent, mergestrat::BlockMerge)
  i1 = 1
  groups = UnitRange{Int64}[]
  for b in mergestrat.possible_breaks
    inext = last_contains_value(parent, b)
    push!(groups, i1:inext)
    i1 = inext + 1
  end
  groups
end

function merged_connection(::Type{DirectMerge}, _, inconn, outconn, i_eliminate, newop, strategy, dimidmap)

  newinputids = [inconn.inputids; filter(!=(i_eliminate), outconn.inputids)]
  inwindows2 = deepcopy(outconn.inwindows)

  i_keep = findall(!=(i_eliminate), outconn.inputids)
  addinwindows = replace_dimids.(inwindows2[i_keep], (dimidmap,))
  newinwindows = (inconn.inwindows..., addinwindows...)
  newoutwindows = replace_dimids.(outconn.outwindows, (dimidmap,))
  MwopConnection(newinputids, outconn.outputids, newop, newinwindows, newoutwindows), []
end

function merged_connection(::Type{BlockMerge}, nodegraph, inconn, outconn, i_eliminate, newop, strategy, dimmap)

  inconninwindows, inconnoutwindows = blockwindows_inconn(inconn, strategy, i_eliminate)
  outconninwindows, outconnoutwindows = blockwindows_outconn(outconn, strategy, i_eliminate)

  outconninwindows = replace_dimids.(outconninwindows, (dimmap,))
  outconnoutwindows = replace_dimids.(outconnoutwindows, (dimmap,))

  outinputids = deepcopy(outconn.inputids)
  itos = findall(==(i_eliminate), outinputids)
  outinputids[itos] .= length(nodegraph.nodes) + 1
  newinputids = [inconn.inputids; outinputids]
  newoutputids = [inconn.outputids; outconn.outputids]

  newinwindows = (inconninwindows..., outconninwindows...)
  newoutwindows = (inconnoutwindows..., outconnoutwindows...)

  newconn = MwopConnection(newinputids, newoutputids, newop, newinwindows, newoutwindows)
  newnode = EmptyInput(nodegraph.nodes[i_eliminate])
  newconn, [newnode]
end

