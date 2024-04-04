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


function merge_operations(inconn,outconn,to_eliminate)
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
        (inconn.f.init...,outconn.f.init...),
        (inconn.f.filters...,outconn.f.filters...),
        outconn.f.finalize,
        outconn.f.buftype,
        outconn.f.outtype,
        inconn.f.allow_threads && outconn.f.allow_threads,
    )
end