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


struct BlockFunctionChain
  funcs
  args
  isel
  transfers::Vector{Pair{Int,Int}} # List of pairs containing the indices of outbuffer-inbuffer pairs to copy data
end
#Constructor to build a length-1 chain
function BlockFunctionChain(f::Union{ElementFunction,BlockFunction},info)
  nin,nout,nlw = info
  funcs = [f]
  args = [ntuple(identity,nin),ntuple(identity,nout)]
  isel = [ntuple(identity,nlw)]
  transfers = []
  BlockFunctionChain(funcs,args,isel,transfers)
end
BlockFunctionChain(op::UserOp,info) = BlockFunctionChain(op.f,info)
BlockFunctionChain(c::BlockFunctionChain,_,_,_) = c
function build_chain(c1::BlockFunctionChain, c2::BlockFunctionChain, dimmap, transfer)
  argspre = maximum(i->maximum.(i),c1.args)
  args = [c1.args;map(i->i.+argspre,c2.args)]
  funcs = [c1.funcs;c2.funcs]
  isel = [c1.isel;map(i->dimmap.(i),c2.isel)]
  inargspre = first(argspre)
  outargspre = last(argspre)
  transfers = copy(c1.transfers)
  for t in c2.transfers
    push!(transfers,(first(t)+outargspre) => (last(t)+inargspre))
  end
  push!(transfers,(first(transfer)+outargspre) => (last(transfer)+inargspre))
  BlockFunctionChain(funcs,args,isel,transfers)
end
maxlr(lw::LoopWindows) = maximum(getloopinds(lw),init=0)
function BlockFunctionChain(conn::MwopConnection)
  nlr = max(maximum(maxlr,conn.inwindows,init=0),maximum(maxlr,conn.outwindows,init=0))
  BlockFunctionChain(conn.f,(length(conn.inwindows),length(conn.outwindows),nlr))
end



function run_block(f::BlockFunctionChain,inow,inbuffers_wrapped,outbuffers_now,threaded)
  
  func1 = first(f.funcs)
  args1 = first(f.args)
  isel1 = first(f.isel)
  inbuffers1 = map(Base.Fix1(getindex,inbuffers_wrapped),first(args1))
  outbuffers1 = map(Base.Fix1(getindex,outbuffers_now),last(args1))
  inow1 = map(Base.Fix1(getindex,inow),isel1)

  run_block(func1,inow1,inbuffers1,outbuffers1,threaded)

  @assert length(f.funcs) == length(f.args) == length(f.isel) == length(f.transfers)+1

  for i in 2:length(f.funcs)
    transfer = f.transfers[i-1]
    for t in transfer
      ob = outbuffers_now[first(t)]
      ib = inbuffers_wrapped[last(t)]
      broadcast!(ob.finalize,ib.a,ob.a)
    end

    func = f.funcs[i]
    args = f.args[i]
    isel = f.isel[i]
    inbuffers = map(Base.Fix1(getindex,inbuffers_wrapped),first(args))
    outbuffers = map(Base.Fix1(getindex,outbuffers_now),last(args))
    inow = map(Base.Fix1(getindex,inow),isel)

    run_block(func,inow,inbuffers,outbuffers,threaded)

  end

end
